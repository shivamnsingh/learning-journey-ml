class Series:

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)

    def mean(self):
        return sum(self.data) / len(self.data)

    def median(self):
        sorted_data = sorted(self.data)
        mid = len(sorted_data) // 2
        if len(sorted_data) % 2 == 0:
            return (sorted_data[mid-1] + sorted_data[mid]) / 2
        else:
            return sorted_data[mid]
    def sum(self):
        return sum(self.data)

    def min(self):
        return min(self.data)

    def max(self):
        return max(self.data)

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
            return Series(result)

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

df = Dataframe(data, columns)
print(df["Weight"].median())
print(df["Weight"].min())
