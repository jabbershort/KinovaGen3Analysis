x0 = sin(q2)
x1 = sin(q3)
x2 = x0*x1
x3 = cos(q2)
x4 = cos(q3)
x5 = x3*x4
x6 = x2 + x5
x7 = x0*x4
x8 = x1*x3
x9 = -x7 + x8
x10 = sin(q4)
x11 = 0.1059*x10
x12 = x11*x7 - x11*x8
x13 = cos(q4)
x14 = x13*x9
x15 = 0.1059*x13
x16 = x10*x9
x17 = 0.062937*x10
x18 = 0.062937*x13
x19 = 0.678*x16
x20 = 1.0e-6*x10
x21 = x20*x7
x22 = 0.008466*x13
x23 = x20*x8
x24 = -x2 - x5
x25 = 0.678*x24
x26 = sin(q5)
x27 = x24*x26
x28 = cos(q5)
x29 = x13*x28
x30 = x29*x9
x31 = -x27 + x30
x32 = x1*x26
x33 = x28*x4
x34 = x13*x33 - x32
x35 = x0*x34
x36 = x26*x4
x37 = x1*x28
x38 = -x13*x37 - x36
x39 = x3*x38
x40 = 0.046429*x10
x41 = x13*x32
x42 = -x33 + x41
x43 = 0.008704*x3
x44 = x13*x36
x45 = -x37 - x44
x46 = 0.008704*x0
x47 = 0.678*x31
x48 = x24*x28
x49 = x14*x26
x50 = -x48 - x49
x51 = 0.678*x50
x52 = sin(q6)
x53 = x32*x52
x54 = cos(q6)
x55 = x10*x54
x56 = x29*x52
x57 = x55 + x56
x58 = x36*x52
x59 = x10*x52
x60 = x59*x9
x61 = x31*x54
x62 = -x60 + x61
x63 = 0.925*x62
x64 = x29*x54 - x59
x65 = -x32*x54 + x4*x64
x66 = x0*x65
x67 = -x1*x64 - x36*x54
x68 = x3*x67
x69 = x55*x9
x70 = x31*x52
x71 = x69 + x70
x72 = 0.925*x71
x73 = -x55 - x56
x74 = x0*(x4*x73 + x53)
x75 = x3*(-x1*x73 + x58)
x76 = -x69 - x70
x77 = 0.925*x76
x78 = x37 + x44
x79 = 0.011402*x0
x80 = x33 - x41
x81 = 0.011402*x3
x82 = 0.5*x62
x83 = 0.5*x76
x84 = 0.5*x48 + 0.5*x49
x85 = 1.674e-5*x2
x86 = 0.65039664*x7
x87 = 0.65039664*x8
x88 = 1.674e-5*x5
x89 = x10**2
x90 = 0.265378986*x9
x91 = x89*x90
x92 = x13**2*x90
x93 = 0.1509075*x9
x94 = x28*x89*x93
x95 = 0.008466*x10
x96 = 1.0e-6*x13
x97 = -x96
x98 = x10*x26
x99 = 1.0e-6*x98
x100 = x10*x28
x101 = 0.046429*x100
x102 = 0.1509075*x13*x31
x103 = 0.046429*x13
x104 = 0.008704*x98
x105 = 0.008704*x100
x106 = 0.011402*x52
x107 = x106*x13
x108 = 0.000281*x54
x109 = x108*x13
x110 = 0.000281*x28
x111 = x110*x59
x112 = 0.011402*x28
x113 = x112*x55
x114 = x13*x52
x115 = 0.058*x114
x116 = x28*x55
x117 = 0.058*x116
x118 = x13*x54
x119 = 0.0615*x118
x120 = x28*x59
x121 = 0.0615*x120
x122 = 0.058*x118
x123 = 0.058*x120
x124 = 0.0615*x114
x125 = 0.0615*x116
x126 = 0.011402*x98
x127 = 0.029798*x13
x128 = x127*x54
x129 = 0.029798*x28
x130 = x129*x59
x131 = 0.000281*x98
x132 = x127*x52
x133 = x129*x55
x134 = x26*x52
x135 = x26*x54
x136 = 0.05365*x71
x137 = 0.029798*x26
numpy.array([[g*(-2.103*x12*x14 - 1.425*x12*x31 - 0.678*x14*(x17*x7 - x17*x8 - 0.008466*x2 - 0.008466*x5) + 1.425*x16*(0.1059*x35 + 0.1059*x39) + 2.103*x16*(x15*x7 - x15*x8) + x19*(1.0e-6*x0*x45 + 1.0e-6*x3*x42 - 0.046429*x35 - 0.046429*x39) + x19*(x18*x7 - x18*x8 - 1.0e-6*x2 - 1.0e-6*x5) - x25*(-x21 + x22*x7 - x22*x8 + x23) - x47*(-x40*x7 + x40*x8 - x42*x43 - x45*x46) - x51*(x21 - x23 + x34*x46 + x38*x43) - 2.781*x6*(0.0064*x7 - 0.0064*x8) - 0.93*x6*(0.01397*x7 - 0.01397*x8) - x63*(-0.0615*x74 - 0.0615*x75) - x63*(0.058*x0*(x4*x57 - x53) + 0.058*x3*(-x1*x57 - x58)) - x72*(-0.058*x66 - 0.058*x68) - x77*(0.0615*x66 + 0.0615*x68) - x82*(-0.029798*x74 - 0.029798*x75 - x78*x79 - x80*x81) - x83*(0.000281*x0*x78 + 0.000281*x3*x80 + 0.029798*x66 + 0.029798*x68) - x84*(x65*x79 + x67*x81 - 0.000281*x74 - 0.000281*x75) - 2.781*x9*(0.0064*x2 + 0.0064*x5) - 0.93*x9*(0.01397*x2 + 0.01397*x5))], [g*(1.784267234*x0 - x102 + x19*(x101 + x99) - x25*(-x95 + x97) + 4.417e-5*x3 - x47*(-x103 - x104) - x51*(-x105 + x96) - x63*(x119 - x121) - x63*(x122 - x123) - x72*(x115 + x117) - x77*(-x124 - x125) - x82*(x126 + x128 - x130) - x83*(-x131 - x132 - x133) - x84*(-x107 + x109 - x111 - x113) + x85 + x86 - x87 + x88 - x91 - x92 - x94)], [g*(x102 + x19*(-x101 - x99) - x25*(x95 + x96) - x47*(x103 + x104) - x51*(x105 + x97) - x63*(-x119 + x121) - x63*(-x122 + x123) - x72*(-x115 - x117) - x77*(x124 + x125) - x82*(-x126 - x128 + x130) - x83*(x131 + x132 + x133) - x84*(x107 - x109 + x111 + x113) - x85 - x86 + x87 - x88 + x91 + x92 + x94)], [g*(0.1105375*x134*x62 - x135*x136 + 0.0568875*x135*x76 + 0.005739948*x14 - 6.78e-7*x16 + x19*(0.046429*x26 - 1.0e-6*x28) + 0.005901312*x26*x50 - 0.005901312*x28*x31 - x82*(-x112 - x137*x52) - x83*(x110 - x137*x54) - x84*(-0.000281*x134 - 0.011402*x135) - x93*x98)], [g*(x136*x52 - 0.119428638*x27 + 0.119428638*x30 - 6.78e-7*x48 - 6.78e-7*x49 - 0.0717865*x52*x76 + 0.1254365*x54*x62 - x84*(x106 - x108))], [g*(-0.005701*x60 + 0.005701*x61 + 0.0001405*x69 + 0.0001405*x70)]])