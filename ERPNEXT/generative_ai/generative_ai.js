// Copyright (c) 2015, Frappe Technologies Pvt. Ltd. and Contributors
// License: GNU General Public License v3. See license.txt

frappe.ui.form.on("Generative AI", {
	onload: function (frm) {

	},
	refresh: function (frm) {

	},
	send: function(frm) {
		frappe.call({
				method: "erpnext.stock.get_item_details.get_bin_details",
				args: {
					item_code: item.item_code,
					warehouse: item.warehouse
				},
				callback: function(res){
					item.blox_stock_qty = res.message.actual_qty;
					me.frm.fields_dict.items.grid.refresh();
				}
			});
	}
});
