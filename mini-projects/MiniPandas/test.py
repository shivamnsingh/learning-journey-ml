
data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", 19, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]
rows = [[20, 60], [20, 22]]
column_index = 0

def average_column(rows, column_index):
    new = []
    for row in rows:
        new.append(row[column_index])
    n = len(new)
    print(sum(new)/n)

def average_column(rows, column_index):
    new = []
    for row in rows:
        new.append(row[column_index])
    n = len(new)
    return sum(new) / n

rows = [[20, 60], [20, 22], [22, 40]]


