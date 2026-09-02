class Series:

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Series({self.data})"

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if not isinstance(other, Series):
            return NotImplemented
        return self.data == other.data

    def mean(self):
        return sum(self.data) / len(self.data)

    def __gt__(self, other):
        result = []
        for value in self.data:
            if value is None:
                result.append(False)
            else:
                result.append(value > other)
        return Series(result)

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

    def __getitem__(self, index):
        print(index)
        return self.data[index]


class Dataframe:

    def __init__(self, data, columns): 
        self.data = data
        self.columns = columns    

def groupby(self, column_name):
    column_no = self.columns.index(column_name)
    d = {}
    for row in self.data:
        
    return d


    def sort_values(self, column_name, ascending=True):
        column_no = self.columns.index(column_name)
        if ascending:
            sorted_data = sorted(
                self.data,
                key=lambda row: (row[column_no] is None, row[column_no])
            )
        else:
            sorted_data = sorted(
                self.data,
                key=lambda row: (row[column_no] is not None, row[column_no]),
                reverse=True
            )
        return Dataframe(sorted_data, self.columns)


    def info(self):
        print(f"<class 'MiniPandas.Dataframe'>")
        print(f"RangeIndex: {len(self.data)} entries, 0 to {len(self.data) - 1}")
        print(f"Data columns (total {len(self.columns)} columns):")
        print(" #   Column        Non-Null Count  Dtype")
        print("---  ------        --------------  -----")
        for column_no, cols in enumerate(self.columns):
            non_null_count = 0
            column_type = None
            for row in self.data:
                if row[column_no] is not None:
                    non_null_count += 1
                    if column_type is None:
                        column_type = type(row[column_no])
            if column_type is not None:
                dtype = column_type.__name__
            else:
                dtype = "unknown"
            print(f"{column_no:<4}{cols:<15}{non_null_count:<16}{dtype}")

    def __repr__(self):
        return f"Dataframe({self.data}, columns={self.columns})"

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if not isinstance(other, Dataframe):
            return NotImplemented
        return self.data == other.data and self.columns == other.columns

    def __getitem__(self, column_name): 
        if isinstance(column_name, str):
            column_no = self.columns.index(column_name)
            result = []
            for i in range(0, len(self.data)):
                result.append(self.data[i][column_no])
            return Series(result)
        elif isinstance(column_name, Series):
            result = []

            for row, condition in zip(self.data, column_name.data):
                if condition:
                    result.append(row)

            return Dataframe(result, self.columns)
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

    def drop(self, column_name):
        column_no = self.columns.index(column_name)
        new_column = self.columns.copy()
        new_column.pop(column_no)
        new_data = []
        for i in self.data:
            copy_row = i.copy()
            copy_row.pop(column_no)
            new_data.append(copy_row)
        return Dataframe(new_data, new_column)

    def __str__(self):
        return str(self.data)

    def head(self, n=5):
        print(self.data[:n])

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))


data = [
    ["Shivam", 20, 60],
    ["Rahul", 21, 55],
    ["Aman", None, 30],
    ["Raj", 22, 40],
    ["Vivek", 20, 22],
    ["Neha", 21, 45]
]
columns = ["Name", "Age", "Weight"]
df = Dataframe(data,columns)
df.groupby("Age")

