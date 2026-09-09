
data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", 19, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]


def average_column(rows, column_index):
    # step 1: collect all values at column_index across every row
    # step 2: return their average