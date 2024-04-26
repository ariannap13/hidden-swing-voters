import json

# read json, file available upon request
with open('../../../data/tweets_vips.json') as f:
    tweets_vips = json.load(f)

for tweet in tweets_vips:
    if "created_at" in tweets_vips[tweet]:
        date = tweets_vips[tweet]["created_at"]
        if date:
            if "T" in date and "Oct" not in date and "Sep" not in date: 
                    tweets_vips[tweet]["created_at"] = date[:10]
            else:
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
                    tweets_vips[tweet]["created_at"] = f"{year}-{month}-{day}"
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
                    tweets_vips[tweet]["created_at"] = f"{year}-{month}-{day}"

# write to json
with open('../../../data/tweets_vips.json', 'w') as f:
    json.dump(tweets_vips, f, indent=4)


# also fix dates in tweets_vips_annotated.json
with open('../../../data/tweets_vips_annotated.json') as f:
    tweets_vips_annotated = json.load(f)

for tweet in tweets_vips_annotated:
    if "created_at" in tweets_vips_annotated[tweet]:
        date = tweets_vips_annotated[tweet]["created_at"]
        if date:
            if "T" in date and "Oct" not in date and "Sep" not in date: 
                    tweets_vips[tweet]["created_at"] = date[:10]
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
                    tweets_vips_annotated[tweet]["created_at"] = f"{year}-{month}-{day}"
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
                    tweets_vips_annotated[tweet]["created_at"] = f"{year}-{month}-{day}"

# write to json
with open('../../../data/tweets_vips_annotated.json', 'w') as f:
    json.dump(tweets_vips_annotated, f, indent=4)


# also fix propaganda_swingers.json, file available upon request
with open('../../../data/propaganda_swingers.json') as f:
    propaganda_swingers = json.load(f)

for tweet in propaganda_swingers:
    list_swingers = propaganda_swingers[tweet]
    # date is element 2 in each sublist in list_swingers
    new_list_swingers = []
    for swinger in list_swingers:
        date = swinger[2]
        if date:
            if "T" in date and "Oct" not in date and "Sep" not in date: 
                    tweets_vips[tweet]["created_at"] = date[:10]
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
                    swinger[2] = f"{year}-{month}-{day}"
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
                    swinger[2] = f"{year}-{month}-{day}"
        new_list_swingers.append(swinger)
    propaganda_swingers[tweet] = new_list_swingers

# write to json
with open('../../../data/propaganda_swingers.json', 'w') as f:
    json.dump(propaganda_swingers, f, indent=4)
        

# also fix propaganda_nonswingers.json, file available upon request
with open('../../../data/propaganda_nonswingers.json') as f:
    propaganda_nonswingers = json.load(f)

for tweet in propaganda_nonswingers:
    list_nonswingers = propaganda_nonswingers[tweet]
    # date is element 2 in each sublist in list_swingers
    new_list_nonswingers = []
    for nonswinger in list_nonswingers:
        date = nonswinger[2]
        if date:
            if "T" in date and "Oct" not in date and "Sep" not in date: 
                    tweets_vips[tweet]["created_at"] = date[:10]
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
                    nonswinger[2] = f"{year}-{month}-{day}"
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
                    nonswinger[2] = f"{year}-{month}-{day}"
        new_list_nonswingers.append(nonswinger)
    propaganda_nonswingers[tweet] = new_list_nonswingers

# write to json
with open('../../../data/propaganda_nonswingers.json', 'w') as f:
    json.dump(propaganda_nonswingers, f, indent=4)