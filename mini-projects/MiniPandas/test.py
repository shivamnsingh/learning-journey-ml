
data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", 19, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]

columns = ["Name", "Age", None]
a = [20, 21, 20, 22, 21, 20]
d = {}
for value in a:
    if value not in d:
        d[value] = []
    d[value].append(value)

print(d)

