def perform_calculation(operation, a, b):
    if operation == "sum":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("除数不能为0")
        return a / b
    else:
        raise ValueError("未知的操作符")