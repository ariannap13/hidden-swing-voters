import json

filename = "tweets_vips" # change according to the file you want to fix

# also fix dates in tweets_vips_annotated.json
with open(f'../../../data/{filename}.json') as f:
    file_tofix = json.load(f)

for tweet in file_tofix:
    if "created_at" in file_tofix[tweet]:
        date = file_tofix[tweet]["created_at"]
        if date:
            if "T" in date and "Oct" not in date and "Sep" not in date: 
                    file_tofix[tweet]["created_at"] = date[:10]
            else:
                date = date.split()
                if len(date) == 6:
                    month = date[1]
                    day = date[2]
                    year = date[5]
                    if month == "Jan":
                        month = "01"
                    elif month == "Feb":
                        month = "02"
                    elif month == "Mar":
                        month = "03"
                    elif month == "Apr":
                        month = "04"
                    elif month == "May":
                        month = "05"
                    elif month == "Jun":
                        month = "06"
                    elif month == "Jul":
                        month = "07"
                    elif month == "Aug":
                        month = "08"
                    elif month == "Sep":
                        month = "09"
                    elif month == "Oct":
                        month = "10"
                    elif month == "Nov":
                        month = "11"
                    elif month == "Dec":
                        month = "12"
                    file_tofix[tweet]["created_at"] = f"{year}-{month}-{day}"
                elif len(date) == 3:
                    month = date[1]
                    day = date[2]
                    year = "2022"
                    if month == "Jan":
                        month = "01"
                    elif month == "Feb":
                        month = "02"
                    elif month == "Mar":
                        month = "03"
                    elif month == "Apr":
                        month = "04"
                    elif month == "May":
                        month = "05"
                    elif month == "Jun":
                        month = "06"
                    elif month == "Jul":
                        month = "07"
                    elif month == "Aug":
                        month = "08"
                    elif month == "Sep":
                        month = "09"
                    elif month == "Oct":
                        month = "10"
                    elif month == "Nov":
                        month = "11"
                    elif month == "Dec":
                        month = "12"
                    file_tofix[tweet]["created_at"] = f"{year}-{month}-{day}"

# write to json
with open(f'../../../data/{filename}.json', 'w') as f:
    json.dump(file_tofix, f, indent=4)