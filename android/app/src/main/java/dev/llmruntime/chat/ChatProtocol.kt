package dev.llmruntime.chat

import org.json.JSONArray
import org.json.JSONObject
import java.io.Reader
import java.net.URI

data class Message(val role: String, val content: String)
data class Completion(val reason: String, val tokens: Int)

/** Successful turns alone are sent back to the stateless server. */
class Conversation {
    val messages = mutableListOf<Message>()
    fun request(user: String): List<Message> {
        require(user.isNotBlank()) { "Enter a message." }
        val result = messages + Message("user", user)
        require(result.size <= 64) { "Conversation is full. Start a new chat." }
        require(result.sumOf { it.content.codePointCount(0, it.content.length) } <= 8192) {
            "Conversation is too long. Start a new chat."
        }
        return result
    }
    fun complete(user: String, answer: String) {
        messages += Message("user", user)
        messages += Message("assistant", answer)
    }
}

fun serverUrl(value: String): String {
    val uri = URI(value.trim())
    require(uri.scheme in listOf("http", "https") && !uri.host.isNullOrBlank() &&
        uri.rawUserInfo == null && uri.rawQuery == null && uri.rawFragment == null &&
        uri.path in listOf("", "/") && (uri.port == -1 || uri.port in 1..65535)) {
        "Use a server address such as http://10.0.2.2:8000 (no path or password)."
    }
    return value.trim().trimEnd('/')
}

fun chatBody(model: String, messages: List<Message>, limit: Int): String {
    require(limit in 1..512) { "Output limit must be 1–512 tokens." }
    return JSONObject().put("model", model).put("messages", JSONArray().apply {
        messages.forEach { put(JSONObject().put("role", it.role).put("content", it.content)) }
    }).put("max_tokens", limit).put("temperature", 0).put("stream", true)
        .put("stream_options", JSONObject().put("include_usage", true)).toString()
}

/** Reader decodes UTF-8 across byte boundaries; only blank-line-terminated SSE is dispatched. */
fun readEvents(reader: Reader, event: (String) -> Boolean) {
    val line = StringBuilder()
    val data = mutableListOf<String>()
    var size = 0
    var afterCr = false
    fun endLine(): Boolean {
        val value = line.toString()
        line.setLength(0)
        if (value.isEmpty()) {
            val keepReading = data.isEmpty() || event(data.joinToString("\n"))
            data.clear(); size = 0
            return keepReading
        }
        if (value == "data" || value.startsWith("data:")) {
            val content = if (value == "data") "" else value.substring(5).removePrefix(" ")
            size += content.length
            data += content
        }
        return true
    }
    while (true) {
        val code = reader.read()
        if (code < 0) return
        val c = code.toChar()
        if (afterCr && c == '\n') { afterCr = false; continue }
        afterCr = c == '\r'
        if (c == '\r' || c == '\n') {
            if (!endLine()) return
        } else line.append(c)
        require(size + line.length <= 1_048_576) { "Server event is too large." }
    }
}

class ChatStream(private val model: String, private val requestId: String, private val onText: (String) -> Unit) {
    private var created: Long? = null
    private var finish: String? = null
    private var tokens: Int? = null
    var result: Completion? = null
        private set
    fun accept(data: String): Boolean {
        if (data == "[DONE]") {
            require(finish != null && tokens != null) { "Answer ended without completion details." }
            result = Completion(finish!!, tokens!!)
            return false
        }
        val chunk = JSONObject(data)
        if (chunk.has("error")) throw IllegalStateException(chunk.getJSONObject("error").optString("message", "Server error"))
        val timestamp = chunk.get("created")
        require(timestamp is Number && timestamp.toLong() >= 0 && timestamp.toDouble() == timestamp.toLong().toDouble())
        if (created == null) created = timestamp.toLong()
        require(chunk.getString("object") == "chat.completion.chunk" &&
            chunk.getString("id") == "chatcmpl-$requestId" && chunk.getString("model") == model &&
            timestamp.toLong() == created) { "Server changed request identity." }
        val choices = chunk.getJSONArray("choices")
        if (choices.length() == 0) {
            require(finish != null && tokens == null) { "Unexpected usage event." }
            val usage = chunk.getJSONObject("usage")
            fun count(key: String): Int {
                val n = usage.get(key)
                require(n is Number && n.toLong() in 0..Int.MAX_VALUE && n.toDouble() == n.toLong().toDouble())
                return n.toInt()
            }
            val prompt = count("prompt_tokens"); val output = count("completion_tokens")
            require(count("total_tokens").toLong() == prompt.toLong() + output) { "Invalid token usage." }
            tokens = output
        } else {
            require(choices.length() == 1 && finish == null && tokens == null) { "Unexpected answer sequence." }
            val choice = choices.getJSONObject(0)
            require(choice.get("index") == 0)
            val delta = choice.getJSONObject("delta")
            require(!delta.has("role") || delta.getString("role") == "assistant")
            if (delta.has("content") && !delta.isNull("content")) {
                val content = delta.get("content")
                require(content is String) { "Server sent invalid text." }
                onText(content)
            }
            if (!choice.isNull("finish_reason")) {
                finish = choice.getString("finish_reason")
                require(finish in listOf("stop", "length")) { "Unexpected completion reason." }
            }
        }
        return true
    }
    fun completed(): Completion = result ?: error("Connection closed before the answer finished.")
}
