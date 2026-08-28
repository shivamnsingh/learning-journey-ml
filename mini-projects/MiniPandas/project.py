class Dataframe:

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def __getitem__(self, column_name):
        if isinstance(column_name, str):
            column_no = self.columns.index(column_name)
            result = []
            for i in range(0, len(self.data)):
                result.append(self.data[i][column_no])
            return result

        else:
            col_pst = []
            for column in column_name:
                col_pst.append(self.columns.index(column))
            result = []
            for row in self.data:
                selected_row = []
                for position in col_pst:
                    selected_row.append(row[position])
                result.append(selected_row)
            return result

    def head(self, n=5):
        print(self.data[:n])

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))


data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", 19, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]

columns = ["Name", "Age", "Weight"]

a = [12,13,14,1,15,16]

b = len(a) // 2
print(b)

print(a[b],a[b-1])