class Value:
    
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        output = Value(self.data + other.data)
        return output

    def __sub__(self, other):
        output = Value(self.data - other.data)
        return output

    def __mul__(self, other):
        output = Value(self.data * other.data)
        return output

    def __div__(self, other):
        output = Value(self.data / other.data)
        return output

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
print(a*b+c)