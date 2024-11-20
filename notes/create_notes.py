import datetime
import os


def main():
    today = datetime.datetime.now()
    submission_date = datetime.datetime(2025, 5, 18)
    time_left = (submission_date - today).days// 7

    # file_name = f"t-{time_left} notes.txt"
    days_until_next_monday = (7 - today.weekday()) % 7
    next_monday = (today + datetime.timedelta(days=days_until_next_monday)).strftime("%d-%m-%Y")
    file_name = f"{next_monday} notes.txt"
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(
                f"submission date: {submission_date},\nweeks left: {time_left}\n\n"
                "progress:\n"
                "\t- \n"
                "next steps:\n"
                "\t- \n"
                "discussion:\n"
                "\t- \n"
                "questions:\n"
                "\t- \n"
            )
    else:
        print(f"t-{time_left} notes.txt already exists")

if __name__ == "__main__":
    main()
