"""
Experiment I: Independent Test Set Benchmark

Creates and evaluates a strictly isolated, manually curated independent benchmark
of 250 diverse customer queries across 7 distribution slices:
  1. Normal In-Domain
  2. Complex Paraphrases
  3. Typographical & Spelling Errors
  4. Short & Ambiguous Inputs
  5. High-Risk Financial & Security Scenarios
  6. Highly Negative / Frustrated Tone
  7. Out-Of-Domain General Queries

Evaluates the completely frozen system without tuning or re-thresholding.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.experiments.common import (
    load_vectorizer_and_models,
    save_csv_dual,
    PROJECT_ROOT,
    RESEARCH_RESULTS_DIR,
    EVAL_DIR
)
from src.intent_risk_map import get_intent_risk_level
from src.sentiment_analyzer import SentimentAnalyzer
from src.ood_detector import OODDetector
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine


def build_curated_independent_benchmark():
    records = []

    # 1. Normal In-Domain Queries (35 queries)
    normal = [
        ("Can I find out where my shipment is currently located?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("I need to see my previous purchases from last month.", "order_history", "IN_DOMAIN", "LOW"),
        ("What time does your downtown retail store open tomorrow?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Can you show me the physical address of your branch?", "store_location", "IN_DOMAIN", "LOW"),
        ("What are the standard delivery fees for residential shipping?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Is the black leather handbag currently in stock?", "availability", "IN_DOMAIN", "LOW"),
        ("Can you display the technical specifications of this monitor?", "product_information", "IN_DOMAIN", "LOW"),
        ("When does the Black Friday discount period start?", "sales_period", "IN_DOMAIN", "LOW"),
        ("How long does express shipping normally take to arrive?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("What are the terms of your 30-day return policy?", "return_policy", "IN_DOMAIN", "LOW"),
        ("What is your official policy regarding customer refunds?", "refund_policy", "IN_DOMAIN", "LOW"),
        ("Which credit cards and payment methods do you accept?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("I would like to leave feedback regarding my recent store visit.", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("I want to submit an idea for a new product feature.", "submit_product_idea", "IN_DOMAIN", "LOW"),
        ("How do I navigate through the mobile shopping application?", "use_app", "IN_DOMAIN", "LOW"),
        ("Add two units of the cotton t-shirt to my basket.", "add_product", "IN_DOMAIN", "LOW"),
        ("Please remove the extra socks from my shopping cart.", "remove_product", "IN_DOMAIN", "LOW"),
        ("Where can I track the courier van delivering my package?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Can I check whether the shoes are available at the mall store?", "availability_in_store", "IN_DOMAIN", "LOW"),
        ("Is this gaming console available on your web store?", "availability_online", "IN_DOMAIN", "LOW"),
        ("What are the weekend operating hours of your customer desk?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("How much does same-day courier service cost?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Could you show me my receipt history for this year?", "order_history", "IN_DOMAIN", "LOW"),
        ("What processor is inside this desktop computer?", "product_information", "IN_DOMAIN", "LOW"),
        ("Can you confirm if delivery takes 3 to 5 business days?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Do you ship orders internationally to Canada?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Are there any clearance discounts running this week?", "sales_period", "IN_DOMAIN", "LOW"),
        ("I need to speak with customer support about general inquiries.", "customer_service", "IN_DOMAIN", "LOW"),
        ("Can I pay using PayPal or Apple Pay at checkout?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("Does your company accept cash on delivery?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("I want to share my thoughts on your new website layout.", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("Where can I download your iOS mobile shopping app?", "use_app", "IN_DOMAIN", "LOW"),
        ("Check the transit progress of package #99214.", "track_delivery", "IN_DOMAIN", "LOW"),
        ("What is the warranty period for this electric kettle?", "product_information", "IN_DOMAIN", "LOW"),
        ("Does your store close early on public holidays?", "store_opening_hours", "IN_DOMAIN", "LOW")
    ]
    for q, i, d, r in normal: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Normal In-Domain"})

    # 2. Complex Paraphrases (35 queries)
    paraphrases = [
        ("Could you kindly inform me about the exact whereabouts of my consignment?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("I am keen to inspect all items acquired in my prior transactions.", "order_history", "IN_DOMAIN", "LOW"),
        ("At what hour do your commercial premises initiate business on Saturdays?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Could you direct me toward your nearest brick-and-mortar outlet?", "store_location", "IN_DOMAIN", "LOW"),
        ("What tariff is levied for expedited parcel conveyance?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Are you still stocking the crimson silk dress in size small?", "availability", "IN_DOMAIN", "LOW"),
        ("Provide me with an exhaustive breakdown of the features of this tablet.", "product_information", "IN_DOMAIN", "LOW"),
        ("When will your semi-annual promotional clearance commence?", "sales_period", "IN_DOMAIN", "LOW"),
        ("What is the estimated chronological duration for delivery to occur?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Kindly elaborate upon the statutory provisions governing item returns.", "return_policy", "IN_DOMAIN", "LOW"),
        ("What formal stipulations apply if I seek reimbursement of funds?", "refund_policy", "IN_DOMAIN", "LOW"),
        ("What transactional mechanisms are supported upon final checkout?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("I desire to register observations regarding the demeanor of staff.", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("I possess a conceptual recommendation for catalog diversification.", "submit_product_idea", "IN_DOMAIN", "LOW"),
        ("Instruct me on the operational mechanics of your smartphone interface.", "use_app", "IN_DOMAIN", "LOW"),
        ("Include an additional pair of wireless earbuds in my order bundle.", "add_product", "IN_DOMAIN", "LOW"),
        ("Eliminate the superfluous charging cable from my current bag.", "remove_product", "IN_DOMAIN", "LOW"),
        ("Ascertain the current transit phase of my ordered luggage.", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Is inventory of the ceramic mug maintained at the regional depot?", "availability_in_store", "IN_DOMAIN", "LOW"),
        ("Can this electronic peripheral be acquired via your digital shop?", "availability_online", "IN_DOMAIN", "LOW"),
        ("What operational schedule governs your support staff during festive periods?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("What expenditure is incurred if I choose rapid dispatch?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("I wish to view archival records of my settled customer invoices.", "order_history", "IN_DOMAIN", "LOW"),
        ("Enumerate the technical attributes and battery capacity of this gadget.", "product_information", "IN_DOMAIN", "LOW"),
        ("How many diurnal cycles are required before the shipment arrives?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Do cross-border freight charges apply for shipments to Mexico?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("When are the seasonal markdown percentages going to take effect?", "sales_period", "IN_DOMAIN", "LOW"),
        ("Connect my conversation with an official customer relations desk.", "customer_service", "IN_DOMAIN", "LOW"),
        ("May I settle the outstanding amount utilizing digital wallet solutions?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("Do you entertain split payments across separate payment cards?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("Allow me to submit a formal evaluation of your service excellence.", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("How may one install the software application upon a mobile handset?", "use_app", "IN_DOMAIN", "LOW"),
        ("Trace the geographical dispatch milestones of my package.", "track_delivery", "IN_DOMAIN", "LOW"),
        ("What guarantees protect this blender against mechanical faults?", "product_information", "IN_DOMAIN", "LOW"),
        ("Are evening shopping hours extended during the winter festival?", "store_opening_hours", "IN_DOMAIN", "LOW")
    ]
    for q, i, d, r in paraphrases: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Complex Paraphrase"})

    # 3. Typographical & Spelling Errors (35 queries)
    typos = [
        ("wher is my prcl trckng numbr?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("hw to view my ordr histry from yestrday?", "order_history", "IN_DOMAIN", "LOW"),
        ("wat time does ur stor open on mondys?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("giv me the loction of nearest outlet", "store_location", "IN_DOMAIN", "LOW"),
        ("how much r delvry shippin fees?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("is the blu t-shrt stil in stck?", "availability", "IN_DOMAIN", "LOW"),
        ("ned mor info abt this prduct pls", "product_information", "IN_DOMAIN", "LOW"),
        ("whn dos the chrismas discunt sale start?", "sales_period", "IN_DOMAIN", "LOW"),
        ("how mny days fr delivry arrivel?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("wats the rturn plicy for cloths?", "return_policy", "IN_DOMAIN", "LOW"),
        ("how do i get a rfnd on canceld item?", "refund_policy", "IN_DOMAIN", "LOW"),
        ("wat paymnt cards do u suport?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("wanna submti fedback bout bad store", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("i hav a prodct idea to shre", "submit_product_idea", "IN_DOMAIN", "LOW"),
        ("how to use the app on my android fone?", "use_app", "IN_DOMAIN", "LOW"),
        ("plz add this itm to my crt", "add_product", "IN_DOMAIN", "LOW"),
        ("remov the secnd shrt from my bsket", "remove_product", "IN_DOMAIN", "LOW"),
        ("trck my disptched parcel statuss", "track_delivery", "IN_DOMAIN", "LOW"),
        ("is the watr bottl availble in shop?", "availability_in_store", "IN_DOMAIN", "LOW"),
        ("can i ordr this onlin right now?", "availability_online", "IN_DOMAIN", "LOW"),
        ("wat r the closng hrs on sundays?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("fast shippng chargs how much?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("show me my old purcheses lst", "order_history", "IN_DOMAIN", "LOW"),
        ("spcs and dimnsions of this tabl", "product_information", "IN_DOMAIN", "LOW"),
        ("estmated delvry date for my pkg?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("intl shippin costs to austrlia?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("wen is next flsh sale live?", "sales_period", "IN_DOMAIN", "LOW"),
        ("i need halp from suport agnt", "customer_service", "IN_DOMAIN", "LOW"),
        ("can i pay wth gpay or credt crd?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("do u accept binance or crupto paymnt?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("feedbck for ur webste servce", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("app is nt working prperly on moble", "use_app", "IN_DOMAIN", "LOW"),
        ("wher is my shipmnt held up?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("is ther any waranty on this devce?", "product_information", "IN_DOMAIN", "LOW"),
        ("stor opning time for tomorow?", "store_opening_hours", "IN_DOMAIN", "LOW")
    ]
    for q, i, d, r in typos: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Typo / Noise"})

    # 4. Short & Ambiguous Queries (35 queries)
    ambiguous = [
        ("Help order", "delivery_issue", "IN_DOMAIN", "LOW"),
        ("Status", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Package", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Receipt", "order_history", "IN_DOMAIN", "LOW"),
        ("Store", "store_location", "IN_DOMAIN", "LOW"),
        ("Hours", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Cost", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("In stock?", "availability", "IN_DOMAIN", "LOW"),
        ("Specs", "product_information", "IN_DOMAIN", "LOW"),
        ("Discount", "sales_period", "IN_DOMAIN", "LOW"),
        ("When arrive?", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Return", "return_policy", "IN_DOMAIN", "LOW"),
        ("Payment", "payment_methods", "IN_DOMAIN", "LOW"),
        ("Feedback", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("App issue", "use_app", "IN_DOMAIN", "LOW"),
        ("Add item", "add_product", "IN_DOMAIN", "LOW"),
        ("Delete item", "remove_product", "IN_DOMAIN", "LOW"),
        ("I have an inquiry.", "customer_service", "IN_DOMAIN", "LOW"),
        ("My parcel is strange.", "delivery_issue", "IN_DOMAIN", "LOW"),
        ("Can you help me?", "customer_service", "IN_DOMAIN", "LOW"),
        ("Something is wrong with my account.", "change_account", "IN_DOMAIN", "HIGH"),
        ("Order problem.", "delivery_issue", "IN_DOMAIN", "LOW"),
        ("Item inquiry.", "product_information", "IN_DOMAIN", "LOW"),
        ("Information please.", "customer_service", "IN_DOMAIN", "LOW"),
        ("Where is it?", "track_delivery", "IN_DOMAIN", "LOW"),
        ("When open?", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("How much?", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Assistance needed.", "customer_service", "IN_DOMAIN", "LOW"),
        ("Can I pay?", "payment_methods", "IN_DOMAIN", "LOW"),
        ("Delivery question.", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Online stock?", "availability_online", "IN_DOMAIN", "LOW"),
        ("In store now?", "availability_in_store", "IN_DOMAIN", "LOW"),
        ("Product details", "product_information", "IN_DOMAIN", "LOW"),
        ("Promotion date", "sales_period", "IN_DOMAIN", "LOW"),
        ("Check shopping bag", "order_history", "IN_DOMAIN", "LOW")
    ]
    for q, i, d, r in ambiguous: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Short & Ambiguous"})

    # 5. High-Risk Financial, Security & Legal Queries (40 queries)
    high_risk = [
        ("I want a complete refund for order #99419 immediately.", "request_refund", "IN_DOMAIN", "HIGH"),
        ("Please issue a refund to my credit card right now.", "request_refund", "IN_DOMAIN", "HIGH"),
        ("I demand my money back for this defective toaster.", "request_refund", "IN_DOMAIN", "HIGH"),
        ("Where is the refund of $85 that you promised last week?", "refund_status", "IN_DOMAIN", "HIGH"),
        ("Check the status of my pending refund claim #3312.", "refund_status", "IN_DOMAIN", "HIGH"),
        ("Why has my refund not been processed after 14 days?", "refund_status", "IN_DOMAIN", "HIGH"),
        ("You double charged my Visa card for this transaction!", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("There is a fraudulent charge of $240 on my account statement.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("My payment failed but money was deducted from my bank.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("Cancel my order #77124 immediately before it gets shipped.", "cancel_order", "IN_DOMAIN", "HIGH"),
        ("I changed my mind, cancel my order and do not ship it.", "cancel_order", "IN_DOMAIN", "HIGH"),
        ("Please cancel all items in my recent order.", "cancel_order", "IN_DOMAIN", "HIGH"),
        ("I forgot my account password and need to reset it now.", "recover_password", "IN_DOMAIN", "HIGH"),
        ("Send a password recovery link to my registered email address.", "recover_password", "IN_DOMAIN", "HIGH"),
        ("Someone unauthorized accessed my account, help me reset password.", "recover_password", "IN_DOMAIN", "HIGH"),
        ("Permanently close my customer account and delete all my data.", "close_account", "IN_DOMAIN", "HIGH"),
        ("I want to terminate my membership and delete my profile.", "close_account", "IN_DOMAIN", "HIGH"),
        ("Shut down my account with immediate effect.", "close_account", "IN_DOMAIN", "HIGH"),
        ("Change my registered primary email address to a new one.", "change_account", "IN_DOMAIN", "HIGH"),
        ("Update the telephone number and residential address on my account.", "change_account", "IN_DOMAIN", "HIGH"),
        ("The package arrived completely crushed and the screen is shattered.", "damaged_delivery", "IN_DOMAIN", "HIGH"),
        ("My delivery was thrown over the gate and the contents are broken.", "damaged_delivery", "IN_DOMAIN", "HIGH"),
        ("The box was delivered torn open and the perfume bottle is smashed.", "damaged_delivery", "IN_DOMAIN", "HIGH"),
        ("The delivery box was empty when I opened it, the phone is missing!", "missing_item", "IN_DOMAIN", "HIGH"),
        ("One pair of shoes is completely missing from my multi-item order.", "missing_item", "IN_DOMAIN", "HIGH"),
        ("You failed to include the power adapter in the package.", "missing_item", "IN_DOMAIN", "HIGH"),
        ("You sent me a red sweater instead of the blue jacket I ordered.", "wrong_item", "IN_DOMAIN", "HIGH"),
        ("The package contains someone else's order, this is the wrong item.", "wrong_item", "IN_DOMAIN", "HIGH"),
        ("I received men's boots instead of women's sneakers.", "wrong_item", "IN_DOMAIN", "HIGH"),
        ("Under GDPR Article 16, I exercise my right to data rectification.", "request_right_to_rectification", "IN_DOMAIN", "HIGH"),
        ("I formally request correction of erroneous personal records in your database.", "request_right_to_rectification", "IN_DOMAIN", "HIGH"),
        ("I demand an immediate chargeback for unauthorized transaction #8812.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("Your billing department made an unauthorized withdrawal from my checking account.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("I need to dispute an incorrect billing surcharge of $50.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("Reverse the payment transaction #5521 immediately.", "payment_issue", "IN_DOMAIN", "HIGH"),
        ("I want to revoke my account credentials and erase payment cards.", "close_account", "IN_DOMAIN", "HIGH"),
        ("My package was stolen from my porch, arrange a refund.", "request_refund", "IN_DOMAIN", "HIGH"),
        ("Refund the expedited shipping fee because delivery was late.", "request_refund", "IN_DOMAIN", "HIGH"),
        ("Cancel my subscription and refund the annual fee.", "cancel_order", "IN_DOMAIN", "HIGH"),
        ("I cannot access my two-factor security token, reset login.", "recover_password", "IN_DOMAIN", "HIGH")
    ]
    for q, i, d, r in high_risk: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "High-Risk Safety"})

    # 6. Aggressive / Frustrated Customer Queries (35 queries)
    angry = [
        ("WHERE ON EARTH IS MY ORDER?! I HAVE WAITED 3 WEEKS AND THIS IS RIDICULOUS!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Your shipping service is an absolute nightmare! Where is my parcel?!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Why the hell are your physical store opening times wrong on the website?!", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("I drove 20 miles and your store was closed! What are your actual hours?!", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Your shipping fees are an outrageous rip-off! Explain these fees!", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("I have sent four messages and nobody answers, pathetic customer service!", "customer_service", "IN_DOMAIN", "LOW"),
        ("Your mobile application crashes every time I tap checkout, fix your garbage app!", "use_app", "IN_DOMAIN", "LOW"),
        ("Is it impossible for you people to keep popular items in stock?!", "availability", "IN_DOMAIN", "LOW"),
        ("Why does it take five days just to process a standard parcel?!", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Your return policy is deliberately confusing and completely anti-consumer!", "return_policy", "IN_DOMAIN", "LOW"),
        ("I am sick and tired of waiting for someone to answer my tracking question!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("This is the worst online shopping experience I have had all year!", "customer_service", "IN_DOMAIN", "LOW"),
        ("Why are you advertising discounts if the checkout code fails?!", "sales_period", "IN_DOMAIN", "LOW"),
        ("Your store location map gave me completely false driving directions!", "store_location", "IN_DOMAIN", "LOW"),
        ("Stop wasting my time with useless automated bots, where is my package?!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("I am completely furious about these unexpected shipping charges!", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Nobody in your company seems to know when products will be restocked!", "availability", "IN_DOMAIN", "LOW"),
        ("Your return window is too short, I demand to know why!", "return_policy", "IN_DOMAIN", "LOW"),
        ("Why did you charge my card before even shipping the items?!", "payment_methods", "IN_DOMAIN", "LOW"),
        ("I demand an explanation for why this item has zero description online!", "product_information", "IN_DOMAIN", "LOW"),
        ("Your customer support representatives are rude and unhelpful!", "customer_service", "IN_DOMAIN", "LOW"),
        ("I am leaving a 1-star review everywhere for this horrific delivery delay!", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("Can any human being at your company tell me where package #8821 is?!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Why are your store staff locking the doors twenty minutes before closing?!", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Your delivery partner dropped my package in the rain! Track delivery #5512!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("I will never purchase anything from this website again in my life!", "submit_feedback", "IN_DOMAIN", "LOW"),
        ("Why is express shipping twice as expensive as last month?!", "shipping_costs", "IN_DOMAIN", "LOW"),
        ("Fix your mobile app immediately, it wiped out my entire saved cart!", "use_app", "IN_DOMAIN", "LOW"),
        ("Where is my purchase history?! The website shows zero orders!", "order_history", "IN_DOMAIN", "LOW"),
        ("Your delivery timeline promised 2 days, today is day 6! Unacceptable!", "delivery_time", "IN_DOMAIN", "LOW"),
        ("Why is your local store out of stock of every single advertised shoe?!", "availability_in_store", "IN_DOMAIN", "LOW"),
        ("You people are running a fraudulent delivery network! Where is my item?!", "track_delivery", "IN_DOMAIN", "LOW"),
        ("Answer my simple question about store timings without giving me excuses!", "store_opening_hours", "IN_DOMAIN", "LOW"),
        ("Why is there no phone number to reach an actual human agent?!", "human_agent", "IN_DOMAIN", "MEDIUM"),
        ("I am filing an official complaint with consumer protection about your service!", "submit_feedback", "IN_DOMAIN", "LOW")
    ]
    for q, i, d, r in angry: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Aggressive / Negative Emotion"})

    # 7. Out-Of-Domain Unrelated Inquiries (35 queries)
    ood = [
        ("What is the weather forecast for Seattle tomorrow afternoon?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Who won the men's singles tennis championship at Wimbledon?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Write a Python function to compute the Fibonacci sequence recursively.", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Can you explain how black holes warp spacetime according to Einstein?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What ingredients are required to bake a classic apple pie?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the capital city of Switzerland?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Tell me an amusing joke about software developers.", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the current price of Bitcoin on cryptocurrency exchanges?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How does the human immune system generate antibodies?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Can you translate 'Where is the library?' into Mandarin Chinese?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How do electric car lithium-ion batteries function?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Who composed the Moonlight Sonata in classical music?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the speed of sound traveling through water at sea level?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How do I cultivate organic tomatoes in a backyard greenhouse?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What are the primary differences between Python and Rust?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Can you write a short rhyming poem about autumn leaves?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What are the symptoms of acute seasonal allergies?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Who was the first president of the United States?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How many planets in our solar system possess planetary rings?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What causes lunar eclipses to appear red from Earth?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Explain how neural networks perform backpropagation during training.", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What are the rules of Texas Hold'em poker?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Give me a five-day tourist itinerary for exploring Rome.", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How do airplanes manage pressurization in the cabin at 35,000 feet?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the difference between nuclear fission and nuclear fusion?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How do I brew cold brew coffee at home without special equipment?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Who wrote the fantasy novel The Hobbit?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the deepest freshwater lake on planet Earth?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Can you explain the philosophical concept of utilitarianism?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How does a telescope collect and focus electromagnetic radiation?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What is the molecular formula of table sugar?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("How do seismologists calculate the epicenter of an earthquake?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What exercises are best for training marathon endurance?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("Can you summarize the plot of George Orwell's 1984?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH"),
        ("What causes rainbows to form after a rain shower?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "HIGH")
    ]
    for q, i, d, r in ood: records.append({"query": q, "expected_intent": i, "domain": d, "risk": r, "slice": "Out-Of-Domain"})

    df = pd.DataFrame(records)
    return df


def run_experiment_i():
    print("=" * 75)
    print("EXPERIMENT I: INDEPENDENT TEST SET EVALUATION")
    print("=" * 75)

    # 1. Construct or load independent dataset
    data_dir = os.path.join(PROJECT_ROOT, "data")
    dataset_path = os.path.join(data_dir, "independent_test_set.csv")
    
    ind_df = build_curated_independent_benchmark()
    ind_df.to_csv(dataset_path, index=False)
    print(f"Independent Benchmark Dataset Created: {len(ind_df)} queries")
    print(f"Saved to: {dataset_path}")
    print("\nSlice Breakdown:")
    print(ind_df["slice"].value_counts().to_string())

    # 2. Load Frozen System (No retraining, no threshold tuning!)
    vectorizer, models = load_vectorizer_and_models()
    calibrated_model = models["Calibrated Linear SVM"]

    sentiment_analyzer = SentimentAnalyzer()
    ood_detector = OODDetector(vectorizer=vectorizer, reference_matrix=None)
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine(confidence_threshold=0.85)

    # 3. Evaluate Frozen System
    print("\nEvaluating Frozen Decision Engine on Independent Dataset...")
    eval_results = []

    for _, row in ind_df.iterrows():
        query = row["query"]
        exp_intent = row["expected_intent"]
        true_domain = row["domain"]
        true_risk = row["risk"]
        slice_name = row["slice"]

        q_vec = vectorizer.transform([query])
        pred_intent = calibrated_model.predict(q_vec)[0]
        probs = calibrated_model.predict_proba(q_vec)[0]
        conf = float(np.max(probs))

        sent_data = sentiment_analyzer.analyze(query)
        ood_data = ood_detector.detect(query, conf, query_vector=q_vec)
        risk_data = risk_engine.evaluate_risk(pred_intent, conf, sent_data, ood_data)
        dec_data = decision_engine.make_decision(query, pred_intent, conf, sent_data, ood_data, risk_data)

        decision = dec_data["decision"]

        # Evaluation flags
        is_intent_correct = (pred_intent == exp_intent) if true_domain == "IN_DOMAIN" else False
        is_ood_correct = (ood_data["domain_status"] == "OUT_OF_DOMAIN") if true_domain == "OUT_OF_DOMAIN" else (ood_data["domain_status"] == "IN_DOMAIN")
        is_automated = (decision == "AUTOMATE")
        is_hr_violation = (is_automated and true_risk == "HIGH")
        is_ood_violation = (is_automated and true_domain == "OUT_OF_DOMAIN")

        eval_results.append({
            "Slice": slice_name,
            "Query": query,
            "Expected Intent": exp_intent,
            "Predicted Intent": pred_intent,
            "Intent Correct": is_intent_correct,
            "Confidence (%)": round(conf * 100, 2),
            "True Domain": true_domain,
            "Detected Domain": ood_data["domain_status"],
            "True Risk": true_risk,
            "Evaluated Risk Level": risk_data["risk_level"],
            "Sentiment": sent_data["sentiment"],
            "Final Decision": decision,
            "Automated": is_automated,
            "High-Risk Violation": is_hr_violation,
            "OOD False Acceptance": is_ood_violation,
            "Reason": dec_data["reason"]
        })

    eval_df = pd.DataFrame(eval_results)

    # 4. Summary Metrics
    total_q = len(eval_df)
    total_auto = eval_df["Automated"].sum()
    total_esc = (eval_df["Final Decision"] == "HUMAN_ESCALATION").sum()
    total_clar = (eval_df["Final Decision"] == "CLARIFY").sum()

    in_domain_mask = (eval_df["True Domain"] == "IN_DOMAIN")
    id_q = in_domain_mask.sum()
    id_intent_acc = eval_df[in_domain_mask]["Intent Correct"].mean() * 100.0

    auto_mask = eval_df["Automated"]
    auto_acc = eval_df[auto_mask & in_domain_mask]["Intent Correct"].sum() / auto_mask.sum() * 100.0 if auto_mask.sum() > 0 else 0.0

    hr_mask = (eval_df["True Risk"] == "HIGH")
    hr_auto_count = eval_df[hr_mask]["Automated"].sum()
    hr_auto_rate = (hr_auto_count / hr_mask.sum()) * 100.0

    ood_mask = (eval_df["True Domain"] == "OUT_OF_DOMAIN")
    ood_fa_count = eval_df[ood_mask]["Automated"].sum()
    ood_fa_rate = (ood_fa_count / ood_mask.sum()) * 100.0

    summary_stats = pd.DataFrame([{
        "Total Benchmark Queries": total_q,
        "In-Domain Queries": id_q,
        "Out-of-Domain Queries": int(ood_mask.sum()),
        "High-Risk Queries": int(hr_mask.sum()),
        "Overall Automation Coverage (%)": round((total_auto / total_q) * 100, 2),
        "In-Domain Intent Accuracy (%)": round(id_intent_acc, 2),
        "Selective Accuracy on Automated Queries (%)": round(auto_acc, 2),
        "Human Escalation Rate (%)": round((total_esc / total_q) * 100, 2),
        "Clarification Rate (%)": round((total_clar / total_q) * 100, 2),
        "High-Risk False Automation Rate (%)": round(hr_auto_rate, 2),
        "High-Risk Automated Count": hr_auto_count,
        "OOD False-Acceptance Rate (%)": round(ood_fa_rate, 2),
        "OOD Falsely Accepted Count": ood_fa_count
    }])

    print("\n" + "=" * 75)
    print("INDEPENDENT BENCHMARK EVALUATION SUMMARY")
    print("=" * 75)
    print(summary_stats.to_string(index=False))

    # Slice Breakdown
    slice_summary = []
    for s_name, grp in eval_df.groupby("Slice"):
        slice_summary.append({
            "Slice": s_name,
            "Count": len(grp),
            "Automated": grp["Automated"].sum(),
            "Clarified": (grp["Final Decision"] == "CLARIFY").sum(),
            "Escalated": (grp["Final Decision"] == "HUMAN_ESCALATION").sum(),
            "Coverage (%)": round(grp["Automated"].mean() * 100, 2),
            "Mean Confidence (%)": round(grp["Confidence (%)"].mean(), 2)
        })
    slice_df = pd.DataFrame(slice_summary)
    print("\nPerformance Across Independent Benchmark Slices:")
    print(slice_df.to_string(index=False))

    save_csv_dual(eval_df, "research_independent_eval.csv")
    save_csv_dual(summary_stats, "research_independent_summary.csv")
    save_csv_dual(slice_df, "research_independent_by_slice.csv")


if __name__ == "__main__":
    run_experiment_i()
