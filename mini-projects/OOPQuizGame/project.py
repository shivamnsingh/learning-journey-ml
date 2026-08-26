class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"my name is {self.name} and i am {self.age} years old")

student1 = Student("Alice", 20)
student1.introduce()
