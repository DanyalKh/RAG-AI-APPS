# Copyright (c) 2015, Frappe Technologies Pvt. Ltd. and Contributors
# License: GNU General Public License v3. See license.txt


import json

import frappe
from frappe.utils import cint
from frappe.utils.nestedset import get_root_of

from erpnext.accounts.doctype.pos_invoice.pos_invoice import get_stock_availability
from erpnext.accounts.doctype.pos_profile.pos_profile import get_child_nodes, get_item_groups
from erpnext.stock.utils import scan_barcode


def search_by_term(search_term, warehouse, price_list):
	result = search_for_serial_or_batch_or_barcode_number(search_term) or {}

	item_code = result.get("item_code", search_term)
	serial_no = result.get("serial_no", "")
	batch_no = result.get("batch_no", "")
	barcode = result.get("barcode", "")

	if not result:
		return

	item_doc = frappe.get_doc("Item", item_code)

	if not item_doc:
		return

	item = {
		"barcode": barcode,
		"batch_no": batch_no,
		"description": item_doc.description,
		"is_stock_item": item_doc.is_stock_item,
		"item_code": item_doc.name,
		"item_image": item_doc.image,
		"item_name": item_doc.item_name,
		"serial_no": serial_no,
		"stock_uom": item_doc.stock_uom,
		"uom": item_doc.stock_uom,
	}

	if barcode:
		barcode_info = next(filter(lambda x: x.barcode == barcode, item_doc.get("barcodes", [])), None)
		if barcode_info and barcode_info.uom:
			uom = next(filter(lambda x: x.uom == barcode_info.uom, item_doc.uoms), {})
			item.update(
				{
					"uom": barcode_info.uom,
					"conversion_factor": uom.get("conversion_factor", 1),
				}
			)

	item_stock_qty, is_stock_item = get_stock_availability(item_code, warehouse)
	item_stock_qty = item_stock_qty // item.get("conversion_factor", 1)
	item.update({"actual_qty": item_stock_qty})

	price = frappe.get_list(
		doctype="Item Price",
		filters={
			"price_list": price_list,
			"item_code": item_code,
			"batch_no": batch_no,
		},
		fields=["uom", "currency", "price_list_rate", "batch_no"],
	)

	def __sort(p):
		p_uom = p.get("uom")

		if p_uom == item.get("uom"):
			return 0
		elif p_uom == item.get("stock_uom"):
			return 1
		else:
			return 2

	# sort by fallback preference. always pick exact uom match if available
	price = sorted(price, key=__sort)

	if len(price) > 0:
		p = price.pop(0)
		item.update(
			{
				"currency": p.get("currency"),
				"price_list_rate": p.get("price_list_rate"),
			}
		)

	return {"items": [item]}


@frappe.whitelist()
def get_items(start, page_length, price_list, item_group, pos_profile, search_term=""):
	warehouse, hide_unavailable_items = frappe.db.get_value(
		"POS Profile", pos_profile, ["warehouse", "hide_unavailable_items"]
	)

	result = []
	condition = ''
	serial_no = ''
	batch_no = ''
	if search_term:
		result = search_by_term(search_term, warehouse, price_list) or []
		if result:
			return result

	if not frappe.db.exists("Item Group", item_group):
		item_group = get_root_of("Item Group")

	# Search GTIN
	if search_term.startswith('01'):
		gtin_details = search_barcode_details(search_term, '[]', '[]', 'Stores - BD', '')
		if len(gtin_details) > 0:
			print(gtin_details)
			item_code = gtin_details.get('item_code')
			batch_no = gtin_details.get('batch_no')
			supplier_batch = gtin_details.get('supplier_batch')
			serial_no = gtin_details.get('serial_no')

			condition = get_conditions(item_code)
	else:
		condition = get_conditions(search_term)

	condition += get_item_group_condition(pos_profile)
	# 010629110912049017261231105D3C2174440374667709-010629110912049017261231105D3C2174440374667706
	# 01062911091204901726123110RE4W2174440374667708
	lft, rgt = frappe.db.get_value("Item Group", item_group, ["lft", "rgt"])

	bin_join_selection, bin_join_condition = "", ""
	if hide_unavailable_items:
		bin_join_selection = ", `tabBin` bin"
		bin_join_condition = (
			"AND bin.warehouse = %(warehouse)s AND bin.item_code = item.name AND bin.actual_qty > 0"
		)

	items_data = frappe.db.sql(
		"""
		SELECT
			item.name AS item_code,
			item.item_name,
			item.description,
			item.stock_uom,
			item.image AS item_image,
			item.is_stock_item
		FROM
			`tabItem` item {bin_join_selection}
		WHERE
			item.disabled = 0
			AND item.has_variants = 0
			AND item.is_sales_item = 1
			AND item.is_fixed_asset = 0
			AND item.item_group in (SELECT name FROM `tabItem Group` WHERE lft >= {lft} AND rgt <= {rgt})
			AND {condition}
			{bin_join_condition}
		ORDER BY
			item.name asc
		LIMIT
			{page_length} offset {start}""".format(
			start=cint(start),
			page_length=cint(page_length),
			lft=cint(lft),
			rgt=cint(rgt),
			condition=condition,
			bin_join_selection=bin_join_selection,
			bin_join_condition=bin_join_condition,
		),
		{"warehouse": warehouse},
		as_dict=1,
	)

	# return (empty) list if there are no results
	if not items_data:
		return result

	for item in items_data:
		uoms = frappe.get_doc("Item", item.item_code).get("uoms", [])

		item.actual_qty, _ = get_stock_availability(item.item_code, warehouse)
		item.uom = item.stock_uom
		item.serial_no = serial_no
		item.batch_no = batch_no

		item_price = frappe.get_all(
			"Item Price",
			fields=["price_list_rate", "currency", "uom", "batch_no"],
			filters={
				"price_list": price_list,
				"item_code": item.item_code,
				"selling": True,
			},
		)

		if not item_price:
			result.append(item)

		for price in item_price:
			uom = next(filter(lambda x: x.uom == price.uom, uoms), {})

			if price.uom != item.stock_uom and uom and uom.conversion_factor:
				item.actual_qty = item.actual_qty // uom.conversion_factor

			result.append(
				{
					**item,
					"price_list_rate": price.get("price_list_rate"),
					"currency": price.get("currency"),
					"uom": price.uom or item.uom,
					"batch_no": batch_no,
				}
			)
	print(result)
	return {"items": result}


@frappe.whitelist()
def search_for_serial_or_batch_or_barcode_number(search_value: str) -> dict[str, str | None]:
	return scan_barcode(search_value)


def get_conditions(search_term):
	condition = "("
	condition += """item.name like {search_term}
		or item.item_name like {search_term}""".format(search_term=frappe.db.escape("%" + search_term + "%"))
	condition += add_search_fields_condition(search_term)
	condition += ")"

	return condition


def add_search_fields_condition(search_term):
	condition = ""
	search_fields = frappe.get_all("POS Search Fields", fields=["fieldname"])
	if search_fields:
		for field in search_fields:
			condition += " or item.`{}` like {}".format(
				field["fieldname"], frappe.db.escape("%" + search_term + "%")
			)
	return condition


def get_item_group_condition(pos_profile):
	cond = "and 1=1"
	item_groups = get_item_groups(pos_profile)
	if item_groups:
		cond = "and item.item_group in (%s)" % (", ".join(["%s"] * len(item_groups)))

	return cond % tuple(item_groups)


@frappe.whitelist()
@frappe.validate_and_sanitize_search_inputs
def item_group_query(doctype, txt, searchfield, start, page_len, filters):
	item_groups = []
	cond = "1=1"
	pos_profile = filters.get("pos_profile")

	if pos_profile:
		item_groups = get_item_groups(pos_profile)

		if item_groups:
			cond = "name in (%s)" % (", ".join(["%s"] * len(item_groups)))
			cond = cond % tuple(item_groups)

	return frappe.db.sql(
		f""" select distinct name from `tabItem Group`
			where {cond} and (name like %(txt)s) limit {page_len} offset {start}""",
		{"txt": "%%%s%%" % txt},
	)


@frappe.whitelist()
def check_opening_entry(user):
	open_vouchers = frappe.db.get_all(
		"POS Opening Entry",
		filters={"user": user, "pos_closing_entry": ["in", ["", None]], "docstatus": 1},
		fields=["name", "company", "pos_profile", "period_start_date"],
		order_by="period_start_date desc",
	)

	return open_vouchers


@frappe.whitelist()
def create_opening_voucher(pos_profile, company, balance_details):
	balance_details = json.loads(balance_details)

	new_pos_opening = frappe.get_doc(
		{
			"doctype": "POS Opening Entry",
			"period_start_date": frappe.utils.get_datetime(),
			"posting_date": frappe.utils.getdate(),
			"user": frappe.session.user,
			"pos_profile": pos_profile,
			"company": company,
		}
	)
	new_pos_opening.set("balance_details", balance_details)
	new_pos_opening.submit()

	return new_pos_opening.as_dict()


@frappe.whitelist()
def get_past_order_list(search_term, status, limit=20):
	fields = ["name", "grand_total", "currency", "customer", "posting_time", "posting_date"]
	invoice_list = []

	if search_term and status:
		invoices_by_customer = frappe.db.get_all(
			"POS Invoice",
			filters={"customer": ["like", f"%{search_term}%"], "status": status},
			fields=fields,
			page_length=limit,
		)
		invoices_by_name = frappe.db.get_all(
			"POS Invoice",
			filters={"name": ["like", f"%{search_term}%"], "status": status},
			fields=fields,
			page_length=limit,
		)

		invoice_list = invoices_by_customer + invoices_by_name
	elif status:
		invoice_list = frappe.db.get_all(
			"POS Invoice", filters={"status": status}, fields=fields, page_length=limit
		)

	return invoice_list


@frappe.whitelist()
def set_customer_info(fieldname, customer, value=""):
	if fieldname == "loyalty_program":
		frappe.db.set_value("Customer", customer, "loyalty_program", value)

	contact = frappe.get_cached_value("Customer", customer, "customer_primary_contact")
	if not contact:
		contact = frappe.db.sql(
			"""
			SELECT parent FROM `tabDynamic Link`
			WHERE
				parenttype = 'Contact' AND
				parentfield = 'links' AND
				link_doctype = 'Customer' AND
				link_name = %s
			""",
			(customer),
			as_dict=1,
		)
		contact = contact[0].get("parent") if contact else None

	if not contact:
		new_contact = frappe.new_doc("Contact")
		new_contact.is_primary_contact = 1
		new_contact.first_name = customer
		new_contact.set("links", [{"link_doctype": "Customer", "link_name": customer}])
		new_contact.save()
		contact = new_contact.name
		frappe.db.set_value("Customer", customer, "customer_primary_contact", contact)

	contact_doc = frappe.get_doc("Contact", contact)
	if fieldname == "email_id":
		contact_doc.set("email_ids", [{"email_id": value, "is_primary": 1}])
		frappe.db.set_value("Customer", customer, "email_id", value)
	elif fieldname == "mobile_no":
		contact_doc.set("phone_nos", [{"phone": value, "is_primary_mobile_no": 1}])
		frappe.db.set_value("Customer", customer, "mobile_no", value)
	contact_doc.save()


@frappe.whitelist()
def get_pos_profile_data(pos_profile):
	pos_profile = frappe.get_doc("POS Profile", pos_profile)
	pos_profile = pos_profile.as_dict()

	_customer_groups_with_children = []
	for row in pos_profile.customer_groups:
		children = get_child_nodes("Customer Group", row.customer_group)
		_customer_groups_with_children.extend(children)

	pos_profile.customer_groups = _customer_groups_with_children
	return pos_profile


@frappe.whitelist()
def search_barcode_details(search_value, serial_list, item_list, store_warehouse, blox_warehouse):
	serial_list_item = json.loads(serial_list)
	item_list = json.loads(item_list)

	item_code = ''
	gtin = ''
	serial_no = ''
	expiry_date = ''
	batch_no = ''
	# if frappe.db.get_value("Batch", search_value, 'name'):
	# 	blox_serial_doc = frappe.get_doc("Batch", search_value)
	# 	if blox_serial_doc:
	# 		return {'item_code': blox_serial_doc.item,
	# 				'expiry': blox_serial_doc.expiry_date,
	# 				'serial_no': '',
	# 				'supplier_batch': blox_serial_doc.supplier_batch,
	# 				'blox_serial_no': '',
	# 				'batch_no': blox_serial_doc.name}
	# 	else:
	# 		return {}
	# elif frappe.db.get_value("Blox Serial No", {'batch_no': search_value}, 'name'):
	# 	if serial_list_item:
	# 		serial_query = 'and name not in (%s)' % ", ".join(["'%s'" % (c) for c in serial_list_item])
	# 	else:
	# 		serial_query = ''
	# 	blox_serial_no = frappe.db.sql("""select * from `tabBlox Serial No`
	# 														where batch_no='%s'
	# 															 and (sales_invoice IS NULL or sales_invoice = '') %s limit 1""" % (
	# 		search_value, serial_query), as_dict=1)
	# 	if blox_serial_no:
	# 		blox_serial_doc = blox_serial_no[0]
	# 		if not blox_serial_doc.sales_invoice:
	# 			return {'item_code': blox_serial_doc.item_code,
	# 					'gtin': blox_serial_doc.gtin,
	# 					'expiry': blox_serial_doc.expiry_date,
	# 					'serial_no': blox_serial_doc.serial_no,
	# 					'supplier_batch': blox_serial_doc.supplier_batch,
	# 					'blox_serial_no': blox_serial_doc.serial_no,
	# 					'batch_no': blox_serial_doc.batch_no}
	# 	else:
	# 		return {}
	# else:
	search_value = search_value.replace(' ', '').replace('(', '').replace(')', '')

	if search_value.startswith('01'):
		gtin_check_code = ['17', '21']
		print("inside search barcode")
		gtin = ''
		serial_no = ''
		batch_no = ''
		expiry = ''
		gtin_end_index = 16
		gtin = search_value[:16]

		raw_text_after_gtin = search_value[gtin_end_index:]

		try:
			year = float(raw_text_after_gtin[2:4])
		except:
			year = raw_text_after_gtin[2:4]

		if raw_text_after_gtin.startswith('11') and year >= 22:
			production_date = raw_text_after_gtin[:8]
			raw_text_after_production = raw_text_after_gtin[8:]

			for i, j in enumerate(raw_text_after_production):
				if i > 0:
					check_code_text = raw_text_after_production[i - 1] + raw_text_after_production[i]

					if raw_text_after_production.startswith('17') and raw_text_after_production[
																	  2:3] >= 22:
						expiry = raw_text_after_production[:8]
						raw_text_after_expiry = raw_text_after_production[8:]
						for x, y in enumerate(raw_text_after_expiry):
							if x > 0:
								check_code_text_serial = raw_text_after_expiry[x - 1] + \
														 raw_text_after_expiry[x]
								if raw_text_after_expiry.startswith('21'):
									if check_code_text_serial == '10':
										serial_no = raw_text_after_expiry[:x - 1]
										batch_no = raw_text_after_expiry[x - 1:].replace('10', '')
										break
								elif raw_text_after_expiry.startswith('10'):
									if check_code_text_serial == '21':
										batch_no = raw_text_after_expiry[:x - 1].replace('10', '')
										serial_no = raw_text_after_expiry[x - 1:]
										break

					elif raw_text_after_production.startswith('10'):
						for x, y in enumerate(raw_text_after_production):
							if x > 0:
								check_code_text = raw_text_after_production[x - 1] + \
												  raw_text_after_production[x]
								if check_code_text == '17' and raw_text_after_production[
															   x + 1:x + 2] >= 22:
									batch_no = raw_text_after_production[:x - 1].replace('10', '')
									expiry = raw_text_after_production[x - 1:x + 7]
									serial_no = raw_text_after_production[x + 7:]
									break
								elif check_code_text == '21':
									batch_no = raw_text_after_production[:x - 1].replace('10', '')
									serial_no_end_text = raw_text_after_production[x - 1:]
									for z, m in enumerate(serial_no_end_text):
										if z > 0:
											check_code_text_serial = serial_no_end_text[z - 1] + \
																	 serial_no_end_text[
																		 z]
											if check_code_text_serial == '17' and serial_no_end_text[
																				  z + 1:z + 2] >= 22:
												serial_no = check_code_text_serial[:z - 1]
												expiry = check_code_text_serial[z - 1:]
												break

					elif raw_text_after_expiry.startswith('21'):
						for x, y in enumerate(raw_text_after_production):
							if x > 0:
								check_code_text = raw_text_after_production[x - 1] + \
												  raw_text_after_production[x]
								if check_code_text == '17' and raw_text_after_production[
															   x + 1:x + 2] >= 22:
									serial_no = raw_text_after_production[:x - 1]
									expiry = raw_text_after_production[x - 1:x + 7]
									batch_no = raw_text_after_production[x + 7:].replace('10', '')
									break
								elif check_code_text == '10':
									serial_no = raw_text_after_production[:x - 1]
									serial_no_end_text = raw_text_after_production[x - 1:]
									for z, m in enumerate(serial_no_end_text):
										if z > 0:
											check_code_text_serial = serial_no_end_text[z - 1] + \
																	 serial_no_end_text[
																		 z]
											if check_code_text_serial == '17' and serial_no_end_text[
																				  z + 1:z + 2] >= 22:
												batch_no = check_code_text_serial[:z - 1].replace('10',
																								  '')
												expiry = check_code_text_serial[z - 1:]
												break

		elif raw_text_after_gtin.startswith('17') and year >= 22 and year <= 30:
			expiry = raw_text_after_gtin[:8]
			raw_text_after_expiry = raw_text_after_gtin[8:]
			for i, j in enumerate(raw_text_after_expiry):
				if i > 0:
					check_code_text = raw_text_after_expiry[i - 1] + raw_text_after_expiry[i]

					if raw_text_after_expiry.startswith('21'):
						if check_code_text == '10':
							serial_no = raw_text_after_expiry[:i - 1]
							batch_no = raw_text_after_expiry[i + 1:]
							break
					elif raw_text_after_expiry.startswith('10'):
						if check_code_text == '21':
							batch_no = raw_text_after_expiry[2:i - 1]
							serial_no = raw_text_after_expiry[i - 1:]
							break
			if not batch_no:
				batch_no = raw_text_after_expiry[2:]

		elif raw_text_after_gtin.startswith('21'):
			for i, j in enumerate(raw_text_after_gtin):
				if i > 0:
					check_code_text = raw_text_after_gtin[i - 1] + raw_text_after_gtin[i]
					try:
						year = float(raw_text_after_gtin[i + 1:i + 3])
						month = float(raw_text_after_gtin[i + 3:i + 5])
					except:
						year = raw_text_after_gtin[i + 1:i + 3]
						month = raw_text_after_gtin[i + 3:i + 5]
					if check_code_text == '17' and year >= 22 and year <= 30 and month <= 12:
						serial_no = raw_text_after_gtin[:i - 1]
						expiry = raw_text_after_gtin[i - 1:i + 7]
						batch_no = raw_text_after_gtin[i + 9:]

						try:
							index = batch_no.find('11')
							if index != -1:
								year = float(batch_no[index + 2:index + 4])
								month = float(batch_no[index + 4:index + 6])
								if year >= 22 and year <= 30 and month <= 12:
									batch_no = batch_no[:index]
						except:
							pass

						break
					elif check_code_text == '10':
						serial_no = raw_text_after_gtin[:i - 1]
						serial_no_end_text = raw_text_after_gtin[i - 1:]

						for x, y in enumerate(serial_no_end_text):
							if x > 0:
								check_code_text_serial = serial_no_end_text[x - 1] + serial_no_end_text[x]
								try:
									year = float(serial_no_end_text[x + 1:x + 3])
								except:
									year = serial_no_end_text[x + 1:x + 3]
								if check_code_text_serial == '17' and year >= 22 and year <= 30:
									batch_no = serial_no_end_text[2:x - 1]
									expiry = serial_no_end_text[x - 1:x + 7]
									try:
										if serial_no_end_text[x + 7:].startswith('10'):
											batch_no = serial_no_end_text[x + 9:]
									except:
										pass
									break
						break
		elif raw_text_after_gtin.startswith('10'):
			for i, j in enumerate(raw_text_after_gtin):
				if i > 0:
					check_code_text = raw_text_after_gtin[i - 1] + raw_text_after_gtin[i]
					check_code_three_text = check_code_text + raw_text_after_gtin[i + 1]
					try:
						year = float(raw_text_after_gtin[i + 1:i + 3])
					except:
						year = raw_text_after_gtin[i + 1:i + 3]
					if check_code_text == '11' and year >= 18 and year <= 30:
						year = raw_text_after_gtin[i + 1:i + 3]
						batch_no = raw_text_after_gtin[2:i - 1]

						production_date = raw_text_after_gtin[i - 1:i + 7]
						raw_text_after_production = raw_text_after_gtin[i + 7:]
						for x, y in enumerate(raw_text_after_production):
							if x > 0:
								check_code_text = raw_text_after_production[x - 1] + \
												  raw_text_after_production[x]
								try:
									year = float(raw_text_after_production[x + 1:x + 3])
								except:
									year = raw_text_after_production[x + 1:x + 3]
								if raw_text_after_production.startswith(
									'17') and year >= 22 and year <= 30:
									expiry = raw_text_after_production[:8]
									serial_no = raw_text_after_production[8:]
									break
								elif raw_text_after_production.startswith('21'):
									for c, d in enumerate(raw_text_after_production):
										if c > 0:
											check_code_text_serial = raw_text_after_production[c - 1] + \
																	 raw_text_after_production[c]
											try:
												year = float(raw_text_after_production[c + 1:c + 3])
											except:
												year = raw_text_after_production[c + 1:c + 3]
											if check_code_text_serial == '17' and year >= 22 and year <= 30:
												expiry = raw_text_after_production[c - 1:c + 7]
												serial_no = raw_text_after_production[:c - 1]
												break

								break
						break
					elif check_code_text == '17' and year >= 22 and year <= 30:
						year = raw_text_after_gtin[i + 1:i + 3]
						batch_no = raw_text_after_gtin[2:i - 1]
						expiry = raw_text_after_gtin[i - 1:i + 7]
						raw_text_after_production = raw_text_after_gtin[i + 7:]
						for x, y in enumerate(raw_text_after_production):
							if x > 0:
								check_code_text = raw_text_after_production[x - 1] + \
												  raw_text_after_production[x]
								try:
									year = float(raw_text_after_production[x + 1:x + 3])
								except:
									year = raw_text_after_production[x + 1:x + 3]
								if raw_text_after_production.startswith('21'):
									for c, d in enumerate(raw_text_after_production):
										if c > 0:
											check_code_text_serial = raw_text_after_production[c - 1] + \
																	 raw_text_after_production[c]
											try:
												year = float(raw_text_after_production[c + 1:c + 3])
											except:
												year = raw_text_after_production[c + 1:c + 3]
											if check_code_text_serial == '17' and year >= 22 and year <= 30:
												expiry = raw_text_after_production[c - 1:c + 7]
												serial_no = raw_text_after_production[:c - 1]
												break

								break
						break
					elif check_code_text == '21' and check_code_three_text != '217':
						batch_no = raw_text_after_gtin[2:i - 1]
						raw_text_after_production = raw_text_after_gtin[i - 1:]
						for x, y in enumerate(raw_text_after_production):
							if x > 0:
								check_code_text = raw_text_after_production[x - 1] + \
												  raw_text_after_production[x]
								try:
									year = float(raw_text_after_production[x + 1:x + 3])
								except:
									year = raw_text_after_production[x + 1:x + 3]
								if raw_text_after_production.startswith(
									'17') and year >= 22 and year <= 30:
									serial_no = raw_text_after_production[:8]
									expiry = raw_text_after_production[8:]
									break
								elif raw_text_after_production.startswith('21'):
									for c, d in enumerate(raw_text_after_production):
										if c > 0:
											check_code_text_serial = raw_text_after_production[c - 1] + \
																	 raw_text_after_production[c]
											try:
												year = float(raw_text_after_production[c + 1:c + 3])
											except:
												year = raw_text_after_production[c + 1:c + 3]
											if check_code_text_serial == '17' and year >= 22 and year <= 30:
												serial_no = raw_text_after_production[c - 1:c + 7]
												expiry = raw_text_after_production[:c - 1]
												break

								break
						break

		item_code = frappe.db.get_value(
		"Item Barcode",
		{"barcode": ['like','%' + gtin[3:] + '%']},
		["barcode", "parent as item_code", "uom"], as_dict=True,)
	# frappe.db.get_value("Item", {'gtin': ['like', '%' + gtin + '%']}, 'item_code')
		print(item_code)
		if item_code:
			expiry_date_first = '20' + expiry[2:4] + '-' + expiry[4:6] + '-' + '01'
			expiry_date_real = '20' + expiry[2:4] + '-' + expiry[4:6] + '-' + expiry[6:8]

			# blox_serial_no = frappe.db.get_value("Item Detail",
			# 									 {'supplier_batch': batch_no, 'item_code': item_code,
			# 									  'expiry_date': expiry_date_first,
			# 									  'warehouse': ['in', [store_warehouse, blox_warehouse]]},
			# 									 'batch_no') or frappe.db.get_value("Item Detail",
			# 																		{
			# 																			'supplier_batch': batch_no,
			# 																			'item_code': item_code,
			# 																			'expiry_date': expiry_date_real,
			# 																			'warehouse': [
			# 																				'in', [
			# 																					store_warehouse,
			# 																					blox_warehouse]]},
			# 																		'batch_no')
			# if blox_serial_no:
			# 	return {'item_code': item_code, 'gtin': gtin,
			# 			'expiry': frappe.db.get_value("Batch", blox_serial_no, 'expiry_date'),
			# 			'serial_no': '',
			# 			'supplier_batch': batch_no, 'blox_serial_no': '', 'batch_no': blox_serial_no}
			# else:
			expiry_date_sql = expiry_date_first or expiry_date_real

			batch_no_from_batch = frappe.db.sql(
				"""select name,expiry_date from `tabBatch` where item = '{}' and expiry_date='{}'  """.format(
					item_code.get("item_code"), expiry_date_sql), as_dict=1)
			print(batch_no_from_batch)
			if batch_no_from_batch:
				batch = batch_no_from_batch[0].name
				return {'item_code': item_code.get("item_code"), 'gtin': gtin,
						'expiry': batch_no_from_batch[0].expiry_date, 'serial_no': serial_no,
						'supplier_batch': batch_no, 'blox_serial_no': '', 'batch_no': batch}
			else:
				batch_list = frappe.db.sql(
					"""select name,expiry_date from `tabBatch` where item='%s' """ % item_code.get("item_code"),
					as_dict=1)
				print(batch_list)
				expiry_name = supplier_batch_name = batch_no_name = ''
				found = False
				for selected_batch in batch_list:
					if selected_batch.name and (
						(selected_batch.name in raw_text_after_gtin) or (
						selected_batch.name in raw_text_after_gtin.upper())):
						expiry_name = selected_batch.expiry_date
						supplier_batch_name = selected_batch.name
						batch_no_name = selected_batch.name

				# 		if supplier_batch_name:
				# 			blox_serial_no = frappe.db.get_value("Item Detail",
				# 												 {
				# 													 'supplier_batch': supplier_batch_name,
				# 													 'item_code': item_code,
				# 													 'expiry_date': expiry_name,
				# 													 'warehouse': ['in',
				# 																   [store_warehouse,
				# 																	blox_warehouse]]},
				# 												 'batch_no')
				# 			if blox_serial_no:
				# 				found = True
				# 				break
				#
				# if found:
				# 	return {'item_code': item_code.get("item_code"), 'gtin': gtin,
				# 			'expiry': expiry_name,
				# 			'serial_no': '',
				# 			'supplier_batch': supplier_batch_name, 'blox_serial_no': '',
				# 			'batch_no': blox_serial_no}
					return {'item_code': item_code.get("item_code"), 'gtin': gtin,
							'expiry': expiry_name,
							'serial_no': serial_no,
							'supplier_batch': supplier_batch_name, 'blox_serial_no': '',
							'batch_no': batch_no_name}
				else:
					return {}
		else:
			return {}
	else:
		return {}
