package dev.llmruntime.chat

/** Display state only. Accepted API history remains owned by Conversation. */
enum class ReplyState { WRITING, COMPLETE, LIMIT, STOPPED, FAILED, EMPTY }

data class ChatTurn(val user: String, var reply: String = "", var state: ReplyState = ReplyState.WRITING)
