use dotenv::dotenv;
use github_flows::{get_octo, GithubLogin::Provided};
use openai_flows::{
    chat::{ChatModel, ChatOptions},
    OpenAIFlows,
};
use slack_flows::{listen_to_channel, send_message_to_channel};
use std::env;
use tiktoken_rs::cl100k_base;
#[no_mangle]
pub fn run() {
    dotenv().ok();

    let trigger_word = env::var("trigger_word").unwrap_or("flows summarize".to_string());
    let slack_workspace = env::var("slack_workspace").unwrap_or("secondstate".to_string());
    let slack_channel = env::var("slack_channel").unwrap_or("github-status".to_string());

    listen_to_channel(&slack_workspace, &slack_channel, |sm| {
        handler(&trigger_word, &slack_workspace, &slack_channel, &sm)
    });
}

#[tokio::main(flavor = "current_thread")]
async fn handler(trigger: &str, workspace: &str, channel: &str, sm: &slack_flows::SlackMessage) {
    let github_login = env::var("github_login").unwrap_or("alabulei1".to_string());
    let github_owner = env::var("github_owner").unwrap_or("alabulei1".to_string());
    let github_repo = env::var("github_repo").unwrap_or("a-test".to_string());

    let number = env::var("number").unwrap().parse::<u64>().unwrap_or(2445);
    send_message_to_channel("ik8", "ch_in", sm.text.clone());

    if sm.text.to_lowercase().contains("flows") {
        let mut openai = OpenAIFlows::new();
        openai.set_retry_times(2);

        let octocrab = get_octo(&Provided(github_login));
        let issues_handle = octocrab.issues(&github_owner, &github_repo);

        let issue = issues_handle.get(number).await.unwrap();
        let issue_creator_name = issue.user.login;
        let mut issue_creator_role = "".to_string();
        issue_creator_role = issue.author_association;
        let issue_title = issue.title;
        let issue_number = issue.number;
        let issue_body = issue.body.unwrap_or("".to_string());
        let issue_url = issue.html_url;
        let labels = issue
            .labels
            .into_iter()
            .map(|lab| lab.name)
            .collect::<Vec<String>>()
            .join(", ");

        let bpe = cl100k_base().unwrap();

        // let mut feed_tokens_map = String::new();
        let mut feed_tokens_map = Vec::new();

        let issue_creator_input = format!("issue creator {issue_creator_name} has role {issue_creator_role}, filed the issue titled {issue_title}, with labels {labels}, posting: {issue_body}");

        let mut tokens = bpe.encode_ordinary(&issue_creator_input);
        feed_tokens_map.append(&mut tokens);
        // feed_tokens_map.push_str(&issue_creator_input);

        match issues_handle.list_comments(number).send().await {
            Ok(pages) => {
                for comment in pages.items {
                    let comment_body = comment.body.unwrap();
                    let commenter = comment.user.login;
                    let commenter_input = format!("{commenter} commented: {comment_body}");
                    let mut tokens = bpe.encode_ordinary(&commenter_input);
                    feed_tokens_map.append(&mut tokens);
                    // feed_tokens_map.push_str(&commenter_input);
                }
            }

            Err(_e) => {}
        }

        let chat_id = format!("Issue#{issue_number}");
        let system = &format!("You are the co-owner of a github repo, you monitor new issues by analyzing the title, body text, labels and its context");

        let co = ChatOptions {
            model: ChatModel::GPT35Turbo,
            restart: true,
            system_prompt: Some(&system),
        };

        let total_tokens_count = feed_tokens_map.len();
        let mut _summary = "".to_string();

        if total_tokens_count > 3000 {
            let mut token_vec = feed_tokens_map;
            let mut map_out = "".to_string();

            while !token_vec.is_empty() {
                let drain_to = std::cmp::min(token_vec.len(), 3000);
                let token_chunk = token_vec.drain(0..drain_to).collect::<Vec<_>>();

                // let text_chunk = token_chunk.join(" ");
                let text_chunk = bpe.decode(token_chunk).unwrap();

                let map_question = format!("The issue is titled {issue_title}, with one chunk of the body text or comment text {text_chunk}. Please focus on the main points of the comment, any proposed solutions, and any consensus or disagreements among the commenters. Please summarize key information in this section.");

                match openai.chat_completion(&chat_id, &map_question, &co).await {
                    Ok(r) => {
                        map_out.push_str(&r.choice);
                        send_message_to_channel("ik8", "ch_mid", r.choice);
                    }
                    Err(_e) => {}
                }
            }

            let reduce_question = format!("{issue_creator_name} with role {issue_creator_role} filed the issue titled {issue_title}, labeled {labels}, here are the key info you extracted from issue body text and comments in chunks {map_out}, please focus on the main points of the comments, any proposed solutions, and any consensus or disagreements among the commenters. Please make a concise summary for this issue to facilitate the next action.");

            match openai
                .chat_completion(&chat_id, &reduce_question, &co)
                .await
            {
                Ok(r) => {
                    _summary = r.choice;
                }
                Err(_e) => {}
            }
            send_message_to_channel("ik8", "ch_out", _summary.clone());
        } else {
            // let issue_body = feed_tokens_map;
            let issue_body = bpe.decode(feed_tokens_map).unwrap();

            let question = format!("{issue_body}, please focus on the main points of the comments, any proposed solutions, and any consensus or disagreements among the commenters. Please make a concise summary for this issue to facilitate the next action.");

            match openai.chat_completion(&chat_id, &question, &co).await {
                Ok(r) => {
                    _summary = r.choice;
                }
                Err(_e) => {}
            }
            send_message_to_channel("ik8", "ch_out", _summary.clone());
        }

        let text = format!("Issue Summary:\n{}\n{}", _summary, issue_url);
        send_message_to_channel(&workspace, &channel, text);
    }
}
