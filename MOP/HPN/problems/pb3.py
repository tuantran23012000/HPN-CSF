def f_1(output):
    return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2+output[0][1] - 12*(output[0][2])) +12)/14

def f_2(output):
    return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2\
        + 8*(output[0][0]) - 44.8*(output[0][1]) + 8*(output[0][2])) +44)/57
def f_3(output):
    return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2 -44.8*(output[0][0])\
         + 8*(output[0][1]) + 8*(output[0][2]))+43.7)/56