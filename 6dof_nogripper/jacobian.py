x0 = cos(q1)
x1 = sin(q1)
x2 = sin(q2)
x3 = 0.41*x2
x4 = cos(q3)
x5 = x2*x4
x6 = x1*x5
x7 = sin(q3)
x8 = cos(q2)
x9 = x1*x8
x10 = x7*x9
x11 = cos(q5)
x12 = 0.1674*x10 - 0.1674*x6
x13 = sin(q5)
x14 = sin(q4)
x15 = x0*x14
x16 = x2*x7
x17 = x1*x16
x18 = x4*x8
x19 = x1*x18
x20 = -x17 - x19
x21 = cos(q4)
x22 = 0.1674*x21
x23 = 0.1674*x15 - x20*x22
x24 = x0*x8
x25 = x0*x16
x26 = 0.3143*x25
x27 = x0*x18
x28 = 0.3143*x27
x29 = 0.1674*x25
x30 = 0.1674*x27
x31 = x24*x7
x32 = x0*x5
x33 = x13*x22
x34 = x1*x21
x35 = x25 + x27
x36 = 0.1674*x14
x37 = 0.1674*x32
x38 = 0.1674*x31
x39 = x1*x14
x40 = 0.1674*x39
x41 = 0.3143*x17
x42 = 0.3143*x19
x43 = 0.1674*x17
x44 = 0.1674*x19
x45 = x0*x21
x46 = 0.3143*x5
x47 = x7*x8
x48 = 0.3143*x47
x49 = 0.1674*x5
x50 = 0.1674*x47
x51 = 0.1674*x16
x52 = 0.1674*x18
x53 = x13*(x51 + x52)
x54 = x49 - x50
x55 = x11*x54
x56 = x13*x21
x57 = q2 - q3
x58 = sin(x57)
x59 = x0*x58
x60 = cos(x57)
x61 = x1*x58
numpy.array([[0.001*x0 - x1*x3 + 0.3143*x10 + x11*x12 + x13*x23 - 0.3143*x6, x11*(x29 + x30) + 0.41*x24 + x26 + x28 - x33*(x31 - x32), x11*(-x29 - x30) - x26 - x28 - x33*(-x31 + x32), x13*(0.1674*x34 + x35*x36), x11*(-x22*x35 + x40) - x13*(x37 - x38), 0], [-x0*x3 - 0.001*x1 + x11*(-x37 + x38) + x13*(-x22*(-x25 - x27) - x40) + 0.3143*x31 - 0.3143*x32, x11*(-x43 - x44) - x33*(-x10 + x6) - x41 - x42 - 0.41*x9, x11*(x43 + x44) - x33*(x10 - x6) + x41 + x42, x13*(x20*x36 + 0.1674*x45), x11*x23 - x12*x13, 0], [0, x11*(-x49 + x50) + x21*x53 - x3 - x46 + x48, x46 - x48 + x55 + x56*(-x51 - x52), -x13*x14*x54, x21*x55 - x53, 0], [0, x1, -x1, -x59, -x15*x60 - x34, -x11*x59 + x13*(-x39 + x45*x60)], [0, x0, -x0, x61, x39*x60 - x45, x11*x61 - x13*(x15 + x34*x60)], [-1, 0, 0, -x60, x14*x58, -x11*x60 - x56*x58]])