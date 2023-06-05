use dotenv::dotenv;
use github_flows::{
    get_octo, listen_to_event,
    octocrab::models::events::payload::{IssueCommentEventAction, IssuesEventAction},
    EventPayload,
    GithubLogin::Default,
};
use openai_flows::{
    chat::{ChatModel, ChatOptions},
    OpenAIFlows,
};
use slack_flows::send_message_to_channel;
use std::env;
use tiktoken_rs::cl100k_base;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() {
    dotenv().ok();

    let github_owner = env::var("github_owner").unwrap_or("alabulei1".to_string());
    let github_repo = env::var("github_repo").unwrap_or("a-test".to_string());
    let trigger_phrase = env::var("trigger_phrase").unwrap_or("issue summarize".to_string());

    listen_to_event(
        &Default,
        &github_owner,
        &github_repo,
        vec!["issues", "issue_comment"],
        |payload| handler(&trigger_phrase, &github_owner, &github_repo, payload),
    )
    .await;
}

async fn handler(trigger: &str, owner: &str, repo: &str, payload: EventPayload) {
    let slack_workspace = env::var("slack_workspace").unwrap_or("secondstate".to_string());
    let slack_channel = env::var("slack_channel").unwrap_or("github-status".to_string());

    // let n_days = env::var("number").unwrap().parse::<u64>().unwrap_or(2445);

    let mut issue = None;
    match payload {
        EventPayload::IssuesEvent(e) => {
            if e.action != IssuesEventAction::Closed
                && e.issue
                    .clone()
                    .body
                    .unwrap_or("".to_string())
                    .contains(&trigger)
            {
                issue = Some(e.issue);
            }
        }

        EventPayload::IssueCommentEvent(e) => {
            if e.action != IssueCommentEventAction::Deleted
                && e.comment
                    .clone()
                    .body
                    .unwrap_or("".to_string())
                    .contains(&trigger)
            {
                issue = Some(e.issue);
            }
        }
        _ => {}
    }

    if let Some(issue) = issue {
        let mut openai = OpenAIFlows::new();
        openai.set_retry_times(3);

        let octocrab = get_octo(&Default);
        let issues_handle = octocrab.issues(owner, repo);

        let issue_creator_name = issue.user.login;
        let mut issue_creator_role = "".to_string();
        issue_creator_role = issue.author_association;
        let issue_number = issue.number;
        let issue_title = issue.title;
        let issue_body = issue.body.unwrap_or("".to_string());
        let issue_url = issue.html_url;
        let labels = issue
            .labels
            .into_iter()
            .map(|lab| lab.name)
            .collect::<Vec<String>>()
            .join(", ");

        let bpe = cl100k_base().unwrap();

        let mut feed_tokens_map = Vec::new();

        let issue_creator_input = format!("User '{issue_creator_name}', who holds the role of '{issue_creator_role}', has submitted an issue titled '{issue_title}', labeled as '{labels}', with the following post: '{issue_body}'.");

        let mut tokens = bpe.encode_ordinary(&issue_creator_input);
        feed_tokens_map.append(&mut tokens);

        match issues_handle.list_comments(issue_number).send().await {
            Ok(pages) => {
                for comment in pages.items {
                    let comment_body = comment.body.unwrap_or("".to_string());
                    let commenter = comment.user.login;
                    let commenter_input = format!("{commenter} commented: {comment_body}");
                    let mut tokens = bpe.encode_ordinary(&commenter_input);
                    feed_tokens_map.append(&mut tokens);
                }
            }

            Err(_e) => {}
        }

        let chat_id = format!("Issue#{issue_number}");
        let system = &format!("As an AI co-owner of a GitHub repository, you are responsible for conducting a comprehensive analysis of GitHub issues. Your analytic focus encompasses distinct elements, including the issue's title, associated labels, body text, the identity of the issue's creator, their role, and the nature of the comments on the issue. Utilizing these data points, your task is to generate a succinct, context-aware summary of the issue.");

        let co = ChatOptions {
            model: ChatModel::GPT35Turbo,
            restart: true,
            system_prompt: Some(system),
        };

        let total_tokens_count = feed_tokens_map.len();
        let mut _summary = "".to_string();

        if total_tokens_count > 2800 {
            let mut token_vec = feed_tokens_map;
            let mut map_out = "".to_string();

            while !token_vec.is_empty() {
                let drain_to = std::cmp::min(token_vec.len(), 2800);
                let token_chunk = token_vec.drain(0..drain_to).collect::<Vec<_>>();

                let text_chunk = bpe.decode(token_chunk).unwrap();

                let map_question = format!("Given the issue titled '{issue_title}' and a particular segment of body or comment text '{text_chunk}', focus on extracting the central arguments, proposed solutions, and instances of agreement or conflict among the participants. Generate an interim summary capturing the essential information in this section. This will be used later to form a comprehensive summary of the entire discussion.");

                match openai.chat_completion(&chat_id, &map_question, &co).await {
                    Ok(r) => {
                        map_out.push_str(&r.choice);
                    }
                    Err(_e) => {}
                }
            }

            let reduce_question = format!("User '{issue_creator_name}', in the role of '{issue_creator_role}', has filed an issue titled '{issue_title}', labeled as '{labels}'. The key information you've extracted from the issue's body text and comments in segmented form are: {map_out}. Concentrate on the principal arguments, suggested solutions, and areas of consensus or disagreement among the participants. From these elements, generate a concise summary of the entire issue to inform the next course of action.");

            match openai
                .chat_completion(&chat_id, &reduce_question, &co)
                .await
            {
                Ok(r) => {
                    _summary = r.choice;
                }
                Err(_e) => {}
            }
        } else {
            let issue_body = bpe.decode(feed_tokens_map).unwrap();

            let question = format!("{issue_body}, concentrate on the principal arguments, suggested solutions, and areas of consensus or disagreement among the participants. From these elements, generate a concise summary of the entire issue to inform the next course of action.");

            match openai.chat_completion(&chat_id, &question, &co).await {
                Ok(r) => {
                    _summary = r.choice;
                }
                Err(_e) => {}
            }
        }

        let text = format!("Issue Summary:\n{}\n{}", _summary, issue_url);
        send_message_to_channel(&slack_workspace, &slack_channel, text);
    }
}
