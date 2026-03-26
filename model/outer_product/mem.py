cout = 64
cin = 16
w = 32
h = 32

weight_size = 16
mp_size = 32

total_adder_num = cout * cin * 6
SRAM_MP = cout * h * w * mp_size
Reg_w = cout * cin * 3 * 3  * weight_size
SRAM_Psum = cout * cin * 3 * w * mp_size
print(f"total_adder_num: {total_adder_num}")
print(f"SRAM_MP: {SRAM_MP/1024/8}KB")
print(f"Reg_w: {Reg_w/1024/8}KB")
print(f"SRAM_Psum: {SRAM_Psum/1024/8}KB")
