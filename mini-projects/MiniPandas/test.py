
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

data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", None, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]
columns = ["Name", "Age", "Weight"]


d = {20: [['Shivam', 20, 60],['Vivek', 20, 22]],
        21: [['Rahul', 21, 55],['Neha', 21, 45]],
        None: [['Aman', None, 30]],
        22: [['Raj', 22, 40]]}

def average_column(rows, column_index):
    new = []
    for row in rows:
        value = row[column_index]
        if isinstance(value, (int, float)):
            new.append(value)
    if len(new) == 0:
        return None
    return sum(new) / len(new)

def mean(self):
    result_rows = []
    for key, rows in self.groups.items():
        new_row = [key]
        for column_index in range(len(self.columns)):
            avg = average_column(rows, column_index)
            if avg is not None:
                new_row.append(avg)
        result_rows.append(new_row)
    return result_rows

