// Copyright (c) 2024, Frappe Technologies and contributors
// For license information, please see license.txt

 frappe.ui.form.on("Chatbot AI", {
 	onload: function (frm) {

	},
	refresh: function (frm) {

	},
	send: function(frm) {
		frappe.call({
				method: "frappe.artificial_intelligence.doctype.chatbot_ai.chatbot_ai.get_response",
				args: {
					user_input: frm.doc.user_input,
					},
			});
	}
 });
