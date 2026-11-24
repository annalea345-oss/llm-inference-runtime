package dev.llmruntime.chat

import android.os.Handler
import android.os.Looper
import androidx.lifecycle.ViewModel
import java.util.concurrent.Executors

/** State survives Activity recreation, but never persists chat text to disk. */
class ChatModel : ViewModel() {
    private val main = Handler(Looper.getMainLooper())
    private val network = Executors.newSingleThreadExecutor()
    private val cancellation = Executors.newFixedThreadPool(2)
    private var disposed = false
    private var generation: Generation? = null
    private val conversation = Conversation()
    var changed: (() -> Unit)? = null
    var base = "http://10.0.2.2:8000"
    var model: String? = null
    var busy = false
    var status = "Connect to your Mac to begin."
    val turns = mutableListOf<ChatTurn>()
    var scrollPosition = 0
    var followLatest = true
    var draft = ""
    var limit = "64"
    private fun update(action: () -> Unit) {
        main.post { if (!disposed) { action(); changed?.invoke() } }
    }
    fun connect(value: String) {
        if (busy) return
        val address = try { serverUrl(value) } catch (e: Exception) {
            status = e.message ?: "Invalid address"; changed?.invoke(); return
        }
        if (base != address) { conversation.messages.clear(); turns.clear(); scrollPosition = 0 }
        base = address; model = null; busy = true; status = "Connecting…"; changed?.invoke()
        network.execute {
            try {
                val name = discover(address)
                update { model = name; busy = false; status = "Connected · $name" }
            } catch (e: Exception) { update { busy = false; status = "Could not connect: ${e.message}" } }
        }
    }
    fun send(user: String, outputLimit: String) {
        if (busy || model == null) return
        val messages: List<Message>
        val tokens: Int
        try {
            tokens = outputLimit.toIntOrNull() ?: error("Output limit must be 1–512 tokens.")
            require(tokens in 1..512) { "Output limit must be 1–512 tokens." }
            messages = conversation.request(user)
        } catch (e: Exception) { status = e.message ?: "Invalid message"; changed?.invoke(); return }
        limit = outputLimit; draft = ""
        busy = true; status = "Generating…"
        val turn = ChatTurn(user)
        turns += turn
        val job = Generation(base, cancellation)
        generation = job
        val selected = model!!
        changed?.invoke()
        network.execute {
            val answer = StringBuilder()
            try {
                val result = job.run(selected, messages, tokens) { piece ->
                    answer.append(piece)
                    update { if (generation === job && !job.stopped.get()) turn.reply += piece }
                }
                update {
                    if (!job.stopped.get()) {
                        if (answer.isNotBlank()) conversation.complete(user, answer.toString())
                        turn.state = if (answer.isBlank()) ReplyState.EMPTY else if (result.reason == "length") ReplyState.LIMIT else ReplyState.COMPLETE
                        status = "${result.tokens} output tokens · ${if (result.reason == "length") "output limit reached" else "complete"}"
                    } else {
                        status = "Stopped · partial reply excluded from history"
                        turn.state = ReplyState.STOPPED
                    }
                    generation = null; busy = false
                }
            } catch (e: Exception) {
                update {
                    status = if (job.stopped.get()) "Stopped · partial reply excluded from history"
                        else "Request failed: ${e.message}"
                    turn.state = if (job.stopped.get()) ReplyState.STOPPED else ReplyState.FAILED
                    generation = null; busy = false
                }
            }
        }
    }
    fun stop() { generation?.stop(); status = "Stopping…"; changed?.invoke() }
    fun clear() {
        if (busy) return
        conversation.messages.clear(); turns.clear(); scrollPosition = 0; draft = ""
        status = model?.let { "Connected · $it" } ?: "Connect to your Mac to begin."
        changed?.invoke()
    }
    override fun onCleared() {
        disposed = true; changed = null
        generation?.stop()
        network.execute { cancellation.shutdown() }
        network.shutdown()
    }
}
