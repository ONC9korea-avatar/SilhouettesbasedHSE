## 2023.11.07 (화)


* 이 날짜부터 테스트 하는것은 FC 레이어를 수정한 것임
    - FC1 : 1024 -> 2048
    - FC2 : 512 -> 1024
    - network 명 : new_RegressionPCA

## 2023.11.09 (목)
 * test결과로 excel파일 생성하는 코드 생성
 - python 코드 위치 : test_results/excel.py
 - 해당 위치에서 실행하면 test_results/results.xlsx 파일이 생성됨
 - test_results 폴더 내의 폴더 (ex.Oct_30_10:56:46_2023)의 config.yaml과 results_only_number.txt 파일을 읽어서 표로 정리해줌