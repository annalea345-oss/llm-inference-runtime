package dev.llmruntime.chat

import android.os.Bundle
import android.graphics.Color
import android.text.InputType
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.lifecycle.ViewModelProvider

class MainActivity : ComponentActivity() {
    private lateinit var state: ChatModel
    private lateinit var address: EditText
    private lateinit var prompt: EditText
    private lateinit var limit: EditText
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private lateinit var connect: Button
    private lateinit var send: Button
    private lateinit var stop: Button
    private lateinit var clear: Button
    private fun dp(n: Int) = (n * resources.displayMetrics.density).toInt()
    private fun text(value: String, size: Float = 16f) = TextView(this).apply {
        text = value; textSize = size; setTextColor(Color.rgb(27, 40, 56))
        setPadding(0, dp(6), 0, dp(6))
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        state = ViewModelProvider(this)[ChatModel::class.java]
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
        val landscape = resources.configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(245, 247, 250))
            setPadding(dp(20), dp(12), dp(20), dp(12))
        }
        // API 35 draws behind system bars; inset the complete form and keyboard.
        root.setOnApplyWindowInsetsListener { view, insets ->
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars() or android.view.WindowInsets.Type.ime())
                view.setPadding(dp(20) + bars.left, dp(12) + bars.top, dp(20) + bars.right, dp(12) + bars.bottom)
            } else {
                @Suppress("DEPRECATION")
                view.setPadding(dp(20) + insets.systemWindowInsetLeft, dp(12) + insets.systemWindowInsetTop,
                    dp(20) + insets.systemWindowInsetRight, dp(12) + insets.systemWindowInsetBottom)
            }
            insets
        }
        if (!landscape) {
            root.addView(text("Runtime Chat", 26f))
            root.addView(text("Your phone chats. Your Mac runs the model.", 13f))
        }
        address = EditText(this).apply {
            hint = "Server address"; contentDescription = "Server address"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(); setText(state.base); id = View.generateViewId()
        }
        val settings = LinearLayout(this)
        if (landscape) settings.addView(address, LinearLayout.LayoutParams(0, -2, 1f)) else root.addView(address)
        connect = Button(this).apply { text = "Connect"; setOnClickListener { state.connect(address.text.toString()) } }
        settings.addView(connect)
        clear = Button(this).apply { text = "New chat"; setOnClickListener { state.clear(); prompt.setText("") } }
        settings.addView(clear)
        root.addView(settings)
        status = text("", 13f); status.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        root.addView(status)
        transcript = text("", 16f).apply { setTextIsSelectable(true) }
        scroll = ScrollView(this).apply { addView(transcript); isFillViewport = true }
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        if (!landscape) root.addView(text("Output limit (tokens, 1–512)", 12f))
        limit = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER; setSingleLine(); setText(state.limit)
            contentDescription = "Output token limit"
        }
        if (!landscape) root.addView(limit)
        prompt = EditText(this).apply {
            hint = "Message"; contentDescription = "Message"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            maxLines = 3; setText(state.draft)
        }
        root.addView(prompt)
        val actions = LinearLayout(this)
        send = Button(this).apply { text = "Send"; setOnClickListener {
            state.send(prompt.text.toString(), limit.text.toString())
            if (state.busy) {
                prompt.setText(state.draft)
                (getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager)
                    .hideSoftInputFromWindow(prompt.windowToken, 0)
            }
        } }
        stop = Button(this).apply { text = "Stop"; setOnClickListener { state.stop() } }
        actions.addView(send); actions.addView(stop)
        if (landscape) { actions.addView(text("Tokens", 12f)); actions.addView(limit, LinearLayout.LayoutParams(dp(80), -2)) }
        root.addView(actions)
        setContentView(root)
        address.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) { render() }
        })
        state.changed = ::render
        render()
    }
    private fun render() {
        status.text = state.status
        val content = state.transcript.ifEmpty { "Connect, then send your first message.\n\nChat stays in memory until you close the app or start a new chat." }
        if (transcript.text.toString() != content) {
            transcript.text = content
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
        address.isEnabled = !state.busy; connect.isEnabled = !state.busy
        clear.isEnabled = !state.busy; limit.isEnabled = !state.busy
        prompt.isEnabled = !state.busy
        send.isEnabled = !state.busy && state.model != null && address.text.toString().trim().trimEnd('/') == state.base
        stop.isEnabled = state.busy && state.status == "Generating…"
    }
    override fun onDestroy() {
        state.draft = prompt.text.toString(); state.limit = limit.text.toString()
        state.changed = null
        super.onDestroy()
    }
}
