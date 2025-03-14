import csv

def save_to_csv(data, file_name="output.csv"):
    """
    데이터를 CSV 파일로 저장하는 함수

    Args:
        data (list of list): 저장할 데이터 (2차원 리스트 형식)
        file_name (str): 저장할 CSV 파일 이름 (기본값: output.csv)
    """
    try:
        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"데이터가 '{file_name}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

def read_from_csv(file_name):
    """
    CSV 파일에서 데이터를 읽어오는 함수

    Args:
        file_name (str): 읽을 CSV 파일 이름

    Returns:
        list of list: 읽어온 데이터를 담은 2차원 리스트
    """
    try:
        with open(file_name, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        print(f"'{file_name}' 파일에서 데이터를 성공적으로 읽어왔습니다.")
        return data
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생: {e}")
        return 