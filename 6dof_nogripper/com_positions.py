x0 = sin(q1)
x1 = cos(q1)
x2 = sin(q2)
x3 = 0.208207*x2
x4 = cos(q2)
x5 = 3.5e-5*x4
x6 = 0.41*x2
x7 = x1*x6
x8 = sin(q3)
x9 = x2*x8
x10 = x1*x9
x11 = cos(q3)
x12 = x11*x2
x13 = x1*x12
x14 = x4*x8
x15 = x1*x14
x16 = x11*x4
x17 = x1*x16
x18 = -x0*x6
x19 = x0*x9
x20 = x0*x12
x21 = x0*x14
x22 = x0*x16
x23 = 0.41*x4 + 0.2848
x24 = sin(q4)
x25 = x0*x24
x26 = cos(q4)
x27 = x10 + x17
x28 = x26*x27
x29 = x24*x27
x30 = x0*x26
x31 = 0.001*x0 + x7
x32 = x1*x24
x33 = -x19 - x22
x34 = x26*x33
x35 = x24*x33
x36 = x1*x26
x37 = 0.001*x1 + x18
x38 = -x12 + x14
x39 = x26*x38
x40 = 1.0e-6*x39
x41 = x24*x38
x42 = -x25 + x28
x43 = sin(q5)
x44 = 0.046429*x43
x45 = -x13 + x15
x46 = cos(q5)
x47 = 0.046429*x46
x48 = x43*x45
x49 = x42*x46
x50 = 0.3143*x13 - 0.3143*x15 + x31
x51 = -x32 + x34
x52 = x20 - x21
x53 = x43*x52
x54 = x46*x51
x55 = -0.3143*x20 + 0.3143*x21 + x37
x56 = -x16 - x9
x57 = x43*x56
x58 = 0.3143*x16 + x23 + 0.3143*x9
x59 = -x48 + x49
x60 = cos(q6)
x61 = 0.000281*x60
x62 = -x29 - x30
x63 = sin(q6)
x64 = 0.000281*x63
x65 = 0.011402*x60
x66 = 0.135698*x43
x67 = 0.135698*x46
x68 = 0.011402*x63
x69 = -x53 + x54
x70 = -x35 - x36
x71 = x39*x46 - x57
numpy.array([[0.010364*x0 - 2.3e-5*x1, 2.3e-5*x0 + 0.010364*x1, 0.22976], [-0.02429*x0 + x1*x3 + x1*x5, -x0*x3 - x0*x5 - 0.02429*x1, -3.5e-5*x2 + 0.208207*x4 + 0.2848], [0.00857*x0 + 1.8e-5*x10 + 0.076168*x13 - 0.076168*x15 + 1.8e-5*x17 + x7, 0.00857*x1 + x18 - 1.8e-5*x19 - 0.076168*x20 + 0.076168*x21 - 1.8e-5*x22, -1.8e-5*x12 + 1.8e-5*x14 + 0.076168*x16 + x23 + 0.076168*x9], [0.271337*x13 - 0.271337*x15 + 1.0e-6*x25 - 1.0e-6*x28 - 0.008466*x29 - 0.008466*x30 + x31, -0.271337*x20 + 0.271337*x21 + 1.0e-6*x32 - 1.0e-6*x34 - 0.008466*x35 - 0.008466*x36 + x37, 0.271337*x16 + x23 - x40 - 0.008466*x41 + 0.271337*x9], [0.008704*x29 + 0.008704*x30 + x42*x44 + x45*x47 + 1.0e-6*x48 - 1.0e-6*x49 + x50, 0.008704*x35 + 0.008704*x36 + x44*x51 + x47*x52 + 1.0e-6*x53 - 1.0e-6*x54 + x55, x39*x44 - x40*x46 + 0.008704*x41 + x47*x56 + 1.0e-6*x57 + x58], [-x42*x66 - x45*x67 + x50 + x59*x61 - x59*x68 + x62*x64 + x62*x65, -x51*x66 - x52*x67 + x55 + x61*x69 + x64*x70 + x65*x70 - x68*x69, -x39*x66 - x41*x64 - x41*x65 - x56*x67 + x58 + x61*x71 - x68*x71]])
