x0 = sin(q1)
x1 = cos(q1)
x2 = sin(q2)
x3 = cos(q3)
x4 = x2*x3
x5 = x1*x4
x6 = sin(q3)
x7 = cos(q2)
x8 = x6*x7
x9 = x1*x8
x10 = cos(q5)
x11 = -x5 + x9
x12 = x10*x11
x13 = sin(q5)
x14 = sin(q4)
x15 = cos(q4)
x16 = x2*x6
x17 = x3*x7
x18 = x1*x16 + x1*x17
x19 = -x0*x14 + x15*x18
x20 = x13*x19
x21 = x0*x4
x22 = x0*x8
x23 = x21 - x22
x24 = x10*x23
x25 = -x0*x16 - x0*x17
x26 = -x1*x14 + x15*x25
x27 = x13*x26
x28 = -x16 - x17
x29 = x10*x28
x30 = -x4 + x8
x31 = x15*x30
x32 = x13*x31
x33 = sin(q6)
x34 = -x0*x15 - x14*x18
x35 = cos(q6)
x36 = x10*x19 - x11*x13
x37 = -x1*x15 - x14*x25
x38 = x10*x26 - x13*x23
x39 = x14*x30
x40 = x10*x31 - x13*x28
numpy.array([[-0.409*x0 - 0.0756*x12 - 0.0756*x20 - 0.1025*x5 + 0.1025*x9], [-0.409*x1 + 0.1025*x21 - 0.1025*x22 - 0.0756*x24 - 0.0756*x27], [-0.1025*x16 - 0.1025*x17 - 0.0756*x29 - 0.0756*x32 + 0.2848]])
numpy.array([[x33*x34 + x35*x36, x33*x36 - x34*x35, -x12 - x20], [x33*x37 + x35*x38, x33*x38 - x35*x37, -x24 - x27], [-x33*x39 + x35*x40, x33*x40 + x35*x39, -x29 - x32]])
