import re
from datetime import datetime, timedelta

# Date replacements for multiple languages
date_replacements = {
    "today": "today", "tomorrow": "tomorrow", "after tomorrow": "after_tomorrow", "day after tomorrow": "after_tomorrow",
    "next week": "next_week", "this week": "next_week",
    "next month": "next_month", "this month": "next_month",
    "monday": "monday", "tuesday": "tuesday", "wednesday": "wednesday",
    "thursday": "thursday", "friday": "friday", "saturday": "saturday", "sunday": "sunday",
    "اليوم": "today", "هاليوم": "today", "النهارده": "today",
    "بكرة": "tomorrow", "بُكره": "tomorrow", "غدًا": "tomorrow", "باكر": "tomorrow",
    "بعد بكرة": "after_tomorrow", "بعد الغد": "after_tomorrow", "عقب بكرة": "after_tomorrow",
    "بعد يومين": "after_tomorrow", "بعد يوم": "tomorrow",
    "الأسبوع الجاي": "next_week", "الاسبوع الجاي": "next_week", "الأسبوع القادم": "next_week", "هذا الأسبوع": "next_week",
    "الشهر الجاي": "next_month", "الشهر القادم": "next_month", "هذا الشهر": "next_month",
    "الجمعة": "friday", "السبت": "saturday", "الأحد": "sunday", "الاثنين": "monday",
    "الثلاثاء": "tuesday", "الأربعاء": "wednesday", "الخميس": "thursday",
    "aaj": "today", "आज": "today",
    "kal": "tomorrow", "आने वाला कल": "tomorrow", "कल": "tomorrow",
    "parson": "after_tomorrow", "परसों": "after_tomorrow",
    "agle hafte": "next_week", "अगले हफ़्ते": "next_week", "इस हफ़्ते": "next_week",
    "agle mahine": "next_month", "अगले महीने": "next_month", "इस महीने": "next_month",
    "सोमवार": "monday", "मंगलवार": "tuesday", "बुधवार": "wednesday",
    "गुरुवार": "thursday", "शुक्रवार": "friday", "शनिवार": "saturday", "रविवार": "sunday",
    "আজ": "today", "কাল": "tomorrow", "পরশু": "after_tomorrow",
    "পরের সপ্তাহ": "next_week", "এই সপ্তাহ": "next_week",
    "পরের মাস": "next_month", "এই মাস": "next_month",
    "সোমবার": "monday", "মঙ্গলবার": "tuesday", "বুধবার": "wednesday",
    "বৃহস্পতিবার": "thursday", "শুক্রবার": "friday", "শনিবার": "saturday", "রবিবার": "sunday"
}

def get_date_replacements():
    return date_replacements

def resolve_relative_date(date_text: str):
    today = datetime.today()
    text = date_text.strip().lower()

    replacements = {
        "today": "today", "tomorrow": "tomorrow", "after tomorrow": "after_tomorrow", "day after tomorrow": "after_tomorrow",
        "next week": "next_week", "this week": "next_week",
        "next month": "next_month", "this month": "next_month",
        "monday": "monday", "tuesday": "tuesday", "wednesday": "wednesday",
        "thursday": "thursday", "friday": "friday", "saturday": "saturday", "sunday": "sunday",
        "اليوم": "today", "هاليوم": "today", "النهارده": "today",
        "بكرة": "tomorrow", "بُكره": "tomorrow", "غدًا": "tomorrow", "باكر": "tomorrow",
        "بعد بكرة": "after_tomorrow", "بعد الغد": "after_tomorrow", "عقب بكرة": "after_tomorrow",
        "بعد يومين": "after_tomorrow", "بعد يوم": "tomorrow",
        "الأسبوع الجاي": "next_week", "الاسبوع الجاي": "next_week", "الأسبوع القادم": "next_week", "هذا الأسبوع": "next_week",
        "الشهر الجاي": "next_month", "الشهر القادم": "next_month", "هذا الشهر": "next_month",
        "الجمعة": "friday", "السبت": "saturday", "الأحد": "sunday", "الاثنين": "monday",
        "الثلاثاء": "tuesday", "الأربعاء": "wednesday", "الخميس": "thursday",
        "aaj": "today", "आज": "today",
        "kal": "tomorrow", "आने वाला कल": "tomorrow", "कल": "tomorrow",
        "parson": "after_tomorrow", "परसों": "after_tomorrow",
        "agle hafte": "next_week", "अगले हफ़्ते": "next_week", "इस हफ़्ते": "next_week",
        "agle mahine": "next_month", "अगले महीने": "next_month", "इस महीने": "next_month",
        "सोमवार": "monday", "मंगलवार": "tuesday", "बुधवार": "wednesday",
        "गुरुवार": "thursday", "शुक्रवार": "friday", "शनिवार": "saturday", "रविवार": "sunday",
        "আজ": "today", "কাল": "tomorrow", "পরশু": "after_tomorrow",
        "পরের সপ্তাহ": "next_week", "এই সপ্তাহ": "next_week",
        "পরের মাস": "next_month", "এই মাস": "next_month",
        "সোমবার": "monday", "মঙ্গলবার": "tuesday", "বুধবার": "wednesday",
        "বৃহস্পতিবার": "thursday", "শুক্রবার": "friday", "শনিবার": "saturday", "রবিবার": "sunday"
    }

    # Sort keys by length descending to replace longer phrases first
    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
    for k in sorted_keys:
        text = text.replace(k, replacements[k])

    if "after_tomorrow" in text:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "tomorrow" in text:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in text:
        return today.strftime("%Y-%m-%d")
    elif "next_week" in text:
        return (today + timedelta(days=7)).strftime("%Y-%m-%d")
    elif "next_month" in text:
        next_month = (today.month % 12) + 1
        year = today.year + (today.month // 12)
        return today.replace(year=year, month=next_month, day=1).strftime("%Y-%m-%d")

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    for day, index in weekdays.items():
        if day in text:
            days_ahead = (index - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Handle "بعد X أيام" or "after X days" patterns
    match = re.search(r'بعد\s+(\d+)\s+أيام', text)
    if match:
        days = int(match.group(1))
        return (today + timedelta(days=days)).strftime("%Y-%m-%d")

    match = re.search(r'after\s+(\d+)\s+days', text)
    if match:
        days = int(match.group(1))
        return (today + timedelta(days=days)).strftime("%Y-%m-%d")

    return date_text
