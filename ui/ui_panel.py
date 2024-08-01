import panel as pn
import param

pn.extension(design='material')

class ChatBot(param.Parameterized):
    chat_history = param.List([])
    user_input = param.String("")
    theme = param.ObjectSelector(default="light", objects=["light", "dark"])

    def __init__(self, **params):
        super(ChatBot, self).__init__(**params)
        self.chat_history.append(("bot", "Hello! I'm HUBer. How can I help you today?"))

    @param.depends('user_input', watch=True)
    def respond(self):
        if self.user_input:
            self.chat_history.append(("user", self.user_input))
            self.chat_history.append(("bot", "Hello"))
            self.user_input = ""

    @param.depends('chat_history', 'theme')
    def view(self):
        chat_boxes = []
        for role, text in self.chat_history:
            if role == "user":
                box = pn.pane.Markdown(f"**You:** {text}", css_classes=['user-message'])
            else:
                box = pn.pane.Markdown(f"**HUBer:** {text}", css_classes=['bot-message'])
            chat_boxes.append(box)

        theme_switch = pn.widgets.Switch(name="Dark Mode", value=self.theme == "dark")
        theme_switch.link(self, callbacks={'value': lambda x: setattr(self, 'theme', "dark" if x else "light")})

        chat_container = pn.Column(*chat_boxes, scroll=True, css_classes=['chat-container'])

        input_field = pn.widgets.TextInput(value="", placeholder="Type your message here...", css_classes=['full-width-input'])
        send_button = pn.widgets.Button(name="Send", button_type="primary")
        
        def send_message(event):
            self.user_input = input_field.value
            input_field.value = ""
        
        send_button.on_click(send_message)
        input_field.param.watch(send_message, 'value')

        input_row = pn.Row(input_field, send_button, css_classes=['input-row'])

        return pn.Column(
            pn.Row(theme_switch, align='end'),
            chat_container,
            input_row,
            css_classes=[f'{self.theme}-theme', 'main-container']
        )

chat_bot = ChatBot()

instructions = """
# How to use HUBer

1. Type your message in the input field at the bottom of the screen.
2. Press Enter or click the 'Send' button to send your message.
3. HUBer will always respond with "Hello".
4. Your conversation history will be displayed in the main chat window.
5. Scroll up to view earlier messages in the conversation.
6. Use the switch in the top-right corner to toggle between light and dark themes.

Enjoy chatting with HUBer!
"""

template = pn.template.MaterialTemplate(
    title="HUBer Chatbot",
    sidebar=[pn.pane.Markdown(instructions)],
    main=[chat_bot.view],
)

template.servable()

pn.extension(raw_css="""
.bk-root {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.main-container {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    position: relative;
}

.chat-container {
    flex-grow: 1;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 60px;
    height: calc(100vh - 120px);
}

.user-message {
    background-color: #007BFF;
    color: white;
    border-radius: 20px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 70%;
    align-self: flex-end;
    margin-left: auto;
}

.bot-message {
    background-color: #f1f1f1;
    color: black;
    border-radius: 20px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 70%;
    align-self: flex-start;
}

.input-row {
    display: flex;
    padding: 10px;
    background-color: inherit;
    border-top: 1px solid #ddd;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
}

.full-width-input {
    flex-grow: 1;
}

.light-theme {
    background-color: white;
    color: black;
}

.dark-theme {
    background-color: #2a2a2a;
    color: white;
}

.dark-theme .bot-message {
    background-color: #3a3a3a;
    color: white;
}

.dark-theme .chat-container {
    border-color: #3a3a3a;
}

.dark-theme .input-row {
    border-color: #3a3a3a;
}

/* Adjust the main content area to accommodate the full-width input */
.bk-Canvas {
    padding-bottom: 60px !important;
}

/* Ensure the input row is above other elements */
.input-row {
    z-index: 1000;
}
""")