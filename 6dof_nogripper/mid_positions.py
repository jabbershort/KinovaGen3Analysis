x0 = sin(q1)
x1 = cos(q1)
x2 = sin(q2)
x3 = 0.205*x2
x4 = cos(q2)
x5 = 0.41*x2
x6 = x1*x5
x7 = cos(q3)
x8 = x2*x7
x9 = x1*x8
x10 = sin(q3)
x11 = x10*x4
x12 = x1*x11
x13 = -x0*x5
x14 = x0*x8
x15 = x0*x11
x16 = x4*x7
x17 = x10*x2
x18 = 0.41*x4 + 0.2848
x19 = 0.001*x0 + x6
x20 = 0.001*x1 + x13
x21 = sin(q4)
x22 = cos(q4)
x23 = -x0*x21 + x22*(x1*x16 + x1*x17)
x24 = sin(q5)
x25 = 0.05295*x24
x26 = x12 - x9
x27 = cos(q5)
x28 = 0.05295*x27
x29 = -0.3143*x12 + x19 + 0.3143*x9
x30 = -x1*x21 + x22*(-x0*x16 - x0*x17)
x31 = x14 - x15
x32 = -0.3143*x14 + 0.3143*x15 + x20
x33 = -x16 - x17
x34 = x22*(x11 - x8)
x35 = 0.3143*x16 + 0.3143*x17 + x18
x36 = 0.13665*x24
x37 = 0.13665*x27
numpy.array([[-0.0027*x0, -0.0027*x1, 0.2206], [-0.0054*x0 + x1*x3, -x0*x3 - 0.0054*x1, 0.205*x4 + 0.2848], [-0.0022*x0 - 0.1042*x12 + x6 + 0.1042*x9, -0.0022*x1 + x13 - 0.1042*x14 + 0.1042*x15, 0.1042*x16 + 0.1042*x17 + x18], [-0.26135*x12 + x19 + 0.26135*x9, -0.26135*x14 + 0.26135*x15 + x20, 0.26135*x16 + 0.26135*x17 + x18], [-x23*x25 - x26*x28 + x29, -x25*x30 - x28*x31 + x32, -x25*x34 - x28*x33 + x35], [-x23*x36 - x26*x37 + x29, -x30*x36 - x31*x37 + x32, -x33*x37 - x34*x36 + x35]])
