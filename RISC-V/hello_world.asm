.data
    str1: .string "HelloWorldaaa\n"

.text
.globl _start

main:
	j _start

_start:
    j _print

_print:
    la a1 str1
    li a0 1
    li a7 64
    li a2 15
    ecall

    
