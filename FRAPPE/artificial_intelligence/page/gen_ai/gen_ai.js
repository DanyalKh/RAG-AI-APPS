frappe.pages['gen_ai'].on_page_load = function(wrapper) {
	var page = frappe.ui.make_app_page({
		parent: wrapper,
		title: 'Gen AI',
		single_column: true
	});

	$(`
        <div class="input-container">
            <!-- Button will be moved here -->
        </div>
    `).appendTo(page.body);

	window.page = page;
    window.wrapper = wrapper;
    var args  = {
            "parent": wrapper,
            "page": page
            };

    new frappe.pages.GENAI(args);
}

frappe.provide("frappe.pages");
frappe.pages.GENAI = Class.extend({
	init: function(args){
		$.extend(this, args);
		this.make();
	},
	make: function(){
		this.make_wrapper();
		this.make_filters();
	},
	make_wrapper: function(){
		var me = this;
		this.wrapper = $(frappe.render_template("gen_ai"), {}).appendTo(me.page.main);
	},

	make_filters: function(){
		debugger;
		this.init_chat();
	},

	init_chat: function(){
		var me = this;
		frappe.run_serially([
				() => {
						let button_field = me.page.add_field({
							"fieldname": "send",
							"fieldtype": "Button",
							"label": __("Send"),
							"click": function(){
								 me.sendMessage()
							},
						});
				button_field.$wrapper.hide();
				$('.input-container').append(button_field.$wrapper);
				button_field.$wrapper.show();
				}
                    ]);
	},

	// Function to send the message
 	sendMessage() {
 		var me = this;
		let userInput = me.page.main.find('#user-input').val().trim();

		if (userInput === "") {
			return;
		}

		// Add user message to chat box
		let messageElement = $('<div></div>').addClass('message').addClass('user-message').text(userInput);
		me.page.main.find('#chat-box').append(messageElement);

		// Scroll to bottom automatically
		let chatBox = me.page.main.find('#chat-box');
		chatBox.scrollTop(chatBox[0].scrollHeight);

		// Clear input field
		me.page.main.find('#user-input').val('');

		// Simulate bot response
		frappe.call({
				method: "frappe.artificial_intelligence.doctype.chatbot_ai.chatbot_ai.get_response",
				args: {
					user_input: userInput,
					},
				callback: function(res){
						if (res.message) {
							debugger;
							let botResponse = res.message.output
							let messageElement = $('<div></div>').addClass('message').addClass('bot-message').text(botResponse);
							me.page.main.find('#chat-box').append(messageElement);
							// Scroll to bottom automatically
							let chatBox = me.page.main.find('#chat-box');
							chatBox.scrollTop(chatBox[0].scrollHeight);
						}
					 }
			});
//		let botResponse = "This is a simulated response for: " + userInput;
//		let messageElement = $('<div></div>').addClass('message').addClass('bot-message').text(botResponse);
//		me.page.main.find('#chat-box').append(messageElement);
//
//		// Scroll to bottom automatically
//		let chatBox = me.page.main.find('#chat-box');
//		chatBox.scrollTop(chatBox[0].scrollHeight);

	},

});
