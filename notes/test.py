import datetime

today = datetime.datetime.now()

# Calculate the number of days to add to get to the next Monday
days_until_next_monday = (7 - today.weekday()) % 7

# Calculate the date of the next Monday
next_monday = today + datetime.timedelta(days=days_until_next_monday)

# Print the date of the next Monday
print("The date of the next Monday is:", next_monday.date())