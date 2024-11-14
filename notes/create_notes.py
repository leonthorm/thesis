import datetime
import os


def main():

    submission_date = datetime.datetime(2025, 5, 1)
    time_left = (submission_date - datetime.datetime.now()).days// 7

    file_name = f"t-{time_left} notes.txt"

    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(
                f"submission date: {submission_date},\nweeks left: {time_left}\n\nt"
                "progress:\n"
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