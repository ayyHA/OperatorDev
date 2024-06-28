	.arch armv8-a
	.file	"gemm_4x4_like_blas_neon.cpp"
	.text
	.section	.rodata
	.align	3
.LC0:
	.string	"float* cacheMalloc(int)"
	.align	3
.LC1:
	.string	"../src/gemm_4x4_like_blas_neon.cpp"
	.align	3
.LC2:
	.string	"flag == 0"
	.text
	.align	2
	.global	_Z11cacheMalloci
	.type	_Z11cacheMalloci, %function
_Z11cacheMalloci:
.LFB4100:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	mov	x29, sp
	str	w0, [sp, 28]
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [x0]
	str	x1, [sp, 56]
	mov	x1,0
	str	xzr, [sp, 48]
	ldrsw	x0, [sp, 28]
	lsl	x1, x0, 2
	add	x0, sp, 48
	mov	x2, x1
	mov	x1, 64
	bl	posix_memalign
	str	w0, [sp, 44]
	ldr	w0, [sp, 44]
	cmp	w0, 0
	beq	.L2
	adrp	x0, .LC0
	add	x3, x0, :lo12:.LC0
	mov	w2, 88
	adrp	x0, .LC1
	add	x1, x0, :lo12:.LC1
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	__assert_fail
.L2:
	ldr	x0, [sp, 48]
	mov	x1, x0
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp, 56]
	ldr	x3, [x0]
	subs	x2, x2, x3
	mov	x3, 0
	beq	.L4
	bl	__stack_chk_fail
.L4:
	mov	x0, x1
	ldp	x29, x30, [sp], 64
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4100:
	.size	_Z11cacheMalloci, .-_Z11cacheMalloci
	.section	.rodata
	.align	3
.LC3:
	.string	"void gemm_4x4_like_blas_neon(int, int, int, float*, int, float*, int, float*, int)"
	.align	3
.LC4:
	.string	"m%4==0 && n%4==0 && k%4==0"
	.align	3
.LC5:
	.string	"m>0 && n>0 && k>0"
	.text
	.align	2
	.global	_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i
	.type	_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i, %function
_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i:
.LFB4101:
	.cfi_startproc
	stp	x29, x30, [sp, -112]!
	.cfi_def_cfa_offset 112
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	mov	x29, sp
	str	w0, [sp, 60]
	str	w1, [sp, 56]
	str	w2, [sp, 52]
	str	x3, [sp, 40]
	str	w4, [sp, 48]
	str	x5, [sp, 32]
	str	w6, [sp, 28]
	str	x7, [sp, 16]
	ldr	w0, [sp, 60]
	and	w0, w0, 3
	cmp	w0, 0
	bne	.L6
	ldr	w0, [sp, 56]
	and	w0, w0, 3
	cmp	w0, 0
	bne	.L6
	ldr	w0, [sp, 52]
	and	w0, w0, 3
	cmp	w0, 0
	beq	.L28
.L6:
	adrp	x0, .LC3
	add	x3, x0, :lo12:.LC3
	mov	w2, 97
	adrp	x0, .LC1
	add	x1, x0, :lo12:.LC1
	adrp	x0, .LC4
	add	x0, x0, :lo12:.LC4
	bl	__assert_fail
.L28:
	ldr	w0, [sp, 60]
	cmp	w0, 0
	ble	.L8
	ldr	w0, [sp, 56]
	cmp	w0, 0
	ble	.L8
	ldr	w0, [sp, 52]
	cmp	w0, 0
	bgt	.L29
.L8:
	adrp	x0, .LC3
	add	x3, x0, :lo12:.LC3
	mov	w2, 98
	adrp	x0, .LC1
	add	x1, x0, :lo12:.LC1
	adrp	x0, .LC5
	add	x0, x0, :lo12:.LC5
	bl	__assert_fail
.L29:
	ldr	w1, [sp, 60]
	ldr	w0, [sp, 52]
	mul	w0, w1, w0
	bl	_Z11cacheMalloci
	str	x0, [sp, 96]
	ldr	w1, [sp, 52]
	ldr	w0, [sp, 56]
	mul	w0, w1, w0
	bl	_Z11cacheMalloci
	str	x0, [sp, 104]
	str	wzr, [sp, 76]
.L27:
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 60]
	cmp	w1, w0
	bge	.L10
	ldr	w1, [sp, 60]
	ldr	w0, [sp, 76]
	sub	w0, w1, w0
	str	w0, [sp, 64]
	ldr	w0, [sp, 64]
	cmp	w0, 1024
	ble	.L11
	mov	w0, 1024
	str	w0, [sp, 64]
.L11:
	str	wzr, [sp, 84]
.L26:
	ldr	w1, [sp, 84]
	ldr	w0, [sp, 52]
	cmp	w1, w0
	bge	.L12
	ldr	w1, [sp, 52]
	ldr	w0, [sp, 84]
	sub	w0, w1, w0
	str	w0, [sp, 72]
	ldr	w0, [sp, 72]
	cmp	w0, 511
	ble	.L13
	mov	w0, 256
	str	w0, [sp, 72]
	b	.L14
.L13:
	ldr	w0, [sp, 72]
	cmp	w0, 256
	ble	.L14
	ldr	w0, [sp, 72]
	lsr	w1, w0, 31
	add	w0, w1, w0
	asr	w0, w0, 1
	add	w0, w0, 3
	and	w0, w0, -4
	str	w0, [sp, 72]
.L14:
	ldr	w0, [sp, 56]
	str	w0, [sp, 68]
	ldr	w0, [sp, 68]
	cmp	w0, 511
	ble	.L15
	mov	w0, 256
	str	w0, [sp, 68]
	b	.L16
.L15:
	ldr	w0, [sp, 68]
	cmp	w0, 256
	ble	.L16
	ldr	w0, [sp, 68]
	lsr	w1, w0, 31
	add	w0, w1, w0
	asr	w0, w0, 1
	add	w0, w0, 3
	and	w0, w0, -4
	str	w0, [sp, 68]
.L16:
	ldr	w1, [sp, 84]
	ldr	w0, [sp, 28]
	mul	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 32]
	add	x0, x1, x0
	ldr	x4, [sp, 104]
	ldr	w3, [sp, 28]
	mov	x2, x0
	ldr	w1, [sp, 68]
	ldr	w0, [sp, 72]
	bl	_Z5packBiiPfiS_
	ldr	w0, [sp, 76]
	str	w0, [sp, 92]
.L21:
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 64]
	add	w0, w1, w0
	ldr	w1, [sp, 92]
	cmp	w1, w0
	bge	.L17
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 64]
	add	w1, w1, w0
	ldr	w0, [sp, 92]
	sub	w0, w1, w0
	str	w0, [sp, 88]
	ldr	w0, [sp, 88]
	cmp	w0, 11
	ble	.L18
	mov	w0, 12
	str	w0, [sp, 88]
	b	.L19
.L18:
	ldr	w0, [sp, 88]
	cmp	w0, 7
	ble	.L20
	mov	w0, 8
	str	w0, [sp, 88]
	b	.L19
.L20:
	ldr	w0, [sp, 88]
	cmp	w0, 3
	ble	.L19
	mov	w0, 4
	str	w0, [sp, 88]
.L19:
	ldr	w1, [sp, 92]
	ldr	w0, [sp, 48]
	mul	w0, w1, w0
	sxtw	x1, w0
	ldrsw	x0, [sp, 84]
	add	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 40]
	add	x2, x1, x0
	ldr	w1, [sp, 92]
	ldr	w0, [sp, 76]
	sub	w1, w1, w0
	ldr	w0, [sp, 72]
	mul	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 96]
	add	x0, x1, x0
	mov	x4, x0
	ldr	w3, [sp, 48]
	ldr	w1, [sp, 72]
	ldr	w0, [sp, 88]
	bl	_Z5packAiiPfiS_
	ldr	w1, [sp, 92]
	ldr	w0, [sp, 76]
	sub	w1, w1, w0
	ldr	w0, [sp, 72]
	mul	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 96]
	add	x2, x1, x0
	ldr	w1, [sp, 92]
	ldr	w0, [sp, 112]
	mul	w0, w1, w0
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 16]
	add	x0, x1, x0
	ldr	w6, [sp, 112]
	mov	x5, x0
	ldr	x4, [sp, 104]
	mov	x3, x2
	ldr	w2, [sp, 72]
	ldr	w1, [sp, 68]
	ldr	w0, [sp, 88]
	bl	_Z10kernel_4x4iiiPfS_S_i
	ldr	w1, [sp, 92]
	ldr	w0, [sp, 88]
	add	w0, w1, w0
	str	w0, [sp, 92]
	b	.L21
.L17:
	ldr	w0, [sp, 68]
	str	w0, [sp, 80]
.L25:
	ldr	w1, [sp, 80]
	ldr	w0, [sp, 56]
	cmp	w1, w0
	bge	.L22
	ldr	w1, [sp, 56]
	ldr	w0, [sp, 80]
	sub	w0, w1, w0
	str	w0, [sp, 68]
	ldr	w0, [sp, 68]
	cmp	w0, 511
	ble	.L23
	mov	w0, 256
	str	w0, [sp, 68]
	b	.L24
.L23:
	ldr	w0, [sp, 68]
	cmp	w0, 256
	ble	.L24
	ldr	w0, [sp, 68]
	lsr	w1, w0, 31
	add	w0, w1, w0
	asr	w0, w0, 1
	add	w0, w0, 3
	and	w0, w0, -4
	str	w0, [sp, 68]
.L24:
	ldr	w1, [sp, 84]
	ldr	w0, [sp, 28]
	mul	w0, w1, w0
	sxtw	x1, w0
	ldrsw	x0, [sp, 80]
	add	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 32]
	add	x0, x1, x0
	ldr	x4, [sp, 104]
	ldr	w3, [sp, 28]
	mov	x2, x0
	ldr	w1, [sp, 68]
	ldr	w0, [sp, 72]
	bl	_Z5packBiiPfiS_
	ldr	w1, [sp, 76]
	ldr	w0, [sp, 112]
	mul	w0, w1, w0
	sxtw	x1, w0
	ldrsw	x0, [sp, 80]
	add	x0, x1, x0
	lsl	x0, x0, 2
	ldr	x1, [sp, 16]
	add	x0, x1, x0
	ldr	w6, [sp, 112]
	mov	x5, x0
	ldr	x4, [sp, 104]
	ldr	x3, [sp, 96]
	ldr	w2, [sp, 72]
	ldr	w1, [sp, 68]
	ldr	w0, [sp, 64]
	bl	_Z10kernel_4x4iiiPfS_S_i
	ldr	w1, [sp, 80]
	ldr	w0, [sp, 68]
	add	w0, w1, w0
	str	w0, [sp, 80]
	b	.L25
.L22:
	ldr	w1, [sp, 84]
	ldr	w0, [sp, 72]
	add	w0, w1, w0
	str	w0, [sp, 84]
	b	.L26
.L12:
	ldr	w0, [sp, 76]
	add	w0, w0, 1024
	str	w0, [sp, 76]
	b	.L27
.L10:
	ldr	x0, [sp, 104]
	bl	free
	ldr	x0, [sp, 96]
	bl	free
	nop
	ldp	x29, x30, [sp], 112
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4101:
	.size	_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i, .-_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i
	.section	.rodata
	.align	3
.LC6:
	.string	"void packA(int, int, float*, int, float*)"
	.align	3
.LC7:
	.string	"m>0 && k>0 && m%4==0 && k%4==0"
	.text
	.align	2
	.global	_Z5packAiiPfiS_
	.type	_Z5packAiiPfiS_, %function
_Z5packAiiPfiS_:
.LFB4102:
	.cfi_startproc
	stp	x29, x30, [sp, -176]!
	.cfi_def_cfa_offset 176
	.cfi_offset 29, -176
	.cfi_offset 30, -168
	mov	x29, sp
	str	w0, [sp, 44]
	str	w1, [sp, 40]
	str	x2, [sp, 32]
	str	w3, [sp, 28]
	str	x4, [sp, 16]
	ldr	w0, [sp, 44]
	cmp	w0, 0
	ble	.L31
	ldr	w0, [sp, 40]
	cmp	w0, 0
	ble	.L31
	ldr	w0, [sp, 44]
	and	w0, w0, 3
	cmp	w0, 0
	bne	.L31
	ldr	w0, [sp, 40]
	and	w0, w0, 3
	cmp	w0, 0
	beq	.L37
.L31:
	adrp	x0, .LC6
	add	x3, x0, :lo12:.LC6
	mov	w2, 174
	adrp	x0, .LC1
	add	x1, x0, :lo12:.LC1
	adrp	x0, .LC7
	add	x0, x0, :lo12:.LC7
	bl	__assert_fail
.L37:
	ldr	x0, [sp, 32]
	str	x0, [sp, 128]
	ldr	x0, [sp, 16]
	str	x0, [sp, 136]
	str	wzr, [sp, 56]
.L36:
	ldr	w0, [sp, 56]
	add	w0, w0, 3
	ldr	w1, [sp, 44]
	cmp	w1, w0
	ble	.L38
	ldr	x0, [sp, 128]
	str	x0, [sp, 144]
	ldrsw	x0, [sp, 28]
	lsl	x0, x0, 2
	ldr	x1, [sp, 128]
	add	x0, x1, x0
	str	x0, [sp, 152]
	ldr	w0, [sp, 28]
	lsl	w0, w0, 1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 128]
	add	x0, x1, x0
	str	x0, [sp, 160]
	ldr	w1, [sp, 28]
	mov	w0, w1
	lsl	w0, w0, 1
	add	w0, w0, w1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 128]
	add	x0, x1, x0
	str	x0, [sp, 168]
	str	wzr, [sp, 60]
.L35:
	ldr	w0, [sp, 60]
	add	w0, w0, 3
	ldr	w1, [sp, 40]
	cmp	w1, w0
	ble	.L34
	ldr	x0, [sp, 144]
	ldr	s0, [x0]
	str	s0, [sp, 64]
	ldr	x0, [sp, 152]
	ldr	s0, [x0]
	str	s0, [sp, 68]
	ldr	x0, [sp, 160]
	ldr	s0, [x0]
	str	s0, [sp, 72]
	ldr	x0, [sp, 168]
	ldr	s0, [x0]
	str	s0, [sp, 76]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 4]
	str	s0, [sp, 80]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 4]
	str	s0, [sp, 84]
	ldr	x0, [sp, 160]
	ldr	s0, [x0, 4]
	str	s0, [sp, 88]
	ldr	x0, [sp, 168]
	ldr	s0, [x0, 4]
	str	s0, [sp, 92]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 8]
	str	s0, [sp, 96]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 8]
	str	s0, [sp, 100]
	ldr	x0, [sp, 160]
	ldr	s0, [x0, 8]
	str	s0, [sp, 104]
	ldr	x0, [sp, 168]
	ldr	s0, [x0, 8]
	str	s0, [sp, 108]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 12]
	str	s0, [sp, 112]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 12]
	str	s0, [sp, 116]
	ldr	x0, [sp, 160]
	ldr	s0, [x0, 12]
	str	s0, [sp, 120]
	ldr	x0, [sp, 168]
	ldr	s0, [x0, 12]
	str	s0, [sp, 124]
	ldr	x0, [sp, 136]
	ldr	s0, [sp, 64]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 4
	ldr	s0, [sp, 68]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 8
	ldr	s0, [sp, 72]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 12
	ldr	s0, [sp, 76]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 16
	ldr	s0, [sp, 80]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 20
	ldr	s0, [sp, 84]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 24
	ldr	s0, [sp, 88]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 28
	ldr	s0, [sp, 92]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 32
	ldr	s0, [sp, 96]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 36
	ldr	s0, [sp, 100]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 40
	ldr	s0, [sp, 104]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 44
	ldr	s0, [sp, 108]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 48
	ldr	s0, [sp, 112]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 52
	ldr	s0, [sp, 116]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 56
	ldr	s0, [sp, 120]
	str	s0, [x0]
	ldr	x0, [sp, 136]
	add	x0, x0, 60
	ldr	s0, [sp, 124]
	str	s0, [x0]
	ldr	x0, [sp, 144]
	add	x0, x0, 16
	str	x0, [sp, 144]
	ldr	x0, [sp, 152]
	add	x0, x0, 16
	str	x0, [sp, 152]
	ldr	x0, [sp, 160]
	add	x0, x0, 16
	str	x0, [sp, 160]
	ldr	x0, [sp, 168]
	add	x0, x0, 16
	str	x0, [sp, 168]
	ldr	x0, [sp, 136]
	add	x0, x0, 64
	str	x0, [sp, 136]
	ldr	w0, [sp, 60]
	add	w0, w0, 4
	str	w0, [sp, 60]
	b	.L35
.L34:
	ldr	w0, [sp, 28]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 128]
	add	x0, x1, x0
	str	x0, [sp, 128]
	ldr	w0, [sp, 56]
	add	w0, w0, 4
	str	w0, [sp, 56]
	b	.L36
.L38:
	nop
	ldp	x29, x30, [sp], 176
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4102:
	.size	_Z5packAiiPfiS_, .-_Z5packAiiPfiS_
	.align	2
	.global	_Z5packBiiPfiS_
	.type	_Z5packBiiPfiS_, %function
_Z5packBiiPfiS_:
.LFB4103:
	.cfi_startproc
	sub	sp, sp, #160
	.cfi_def_cfa_offset 160
	str	w0, [sp, 28]
	str	w1, [sp, 24]
	str	x2, [sp, 16]
	str	w3, [sp, 12]
	str	x4, [sp]
	ldr	x0, [sp, 16]
	str	x0, [sp, 112]
	str	wzr, [sp, 40]
.L43:
	ldr	w0, [sp, 40]
	add	w0, w0, 3
	ldr	w1, [sp, 28]
	cmp	w1, w0
	ble	.L44
	ldr	w0, [sp, 40]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp]
	add	x0, x1, x0
	str	x0, [sp, 120]
	ldr	x0, [sp, 112]
	str	x0, [sp, 128]
	ldrsw	x0, [sp, 12]
	lsl	x0, x0, 2
	ldr	x1, [sp, 112]
	add	x0, x1, x0
	str	x0, [sp, 136]
	ldr	w0, [sp, 12]
	lsl	w0, w0, 1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 112]
	add	x0, x1, x0
	str	x0, [sp, 144]
	ldr	w1, [sp, 12]
	mov	w0, w1
	lsl	w0, w0, 1
	add	w0, w0, w1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 112]
	add	x0, x1, x0
	str	x0, [sp, 152]
	str	wzr, [sp, 44]
.L42:
	ldr	w0, [sp, 44]
	add	w0, w0, 3
	ldr	w1, [sp, 24]
	cmp	w1, w0
	ble	.L41
	ldr	x0, [sp, 128]
	ldr	s0, [x0]
	str	s0, [sp, 48]
	ldr	x0, [sp, 128]
	ldr	s0, [x0, 4]
	str	s0, [sp, 52]
	ldr	x0, [sp, 128]
	ldr	s0, [x0, 8]
	str	s0, [sp, 56]
	ldr	x0, [sp, 128]
	ldr	s0, [x0, 12]
	str	s0, [sp, 60]
	ldr	x0, [sp, 136]
	ldr	s0, [x0]
	str	s0, [sp, 64]
	ldr	x0, [sp, 136]
	ldr	s0, [x0, 4]
	str	s0, [sp, 68]
	ldr	x0, [sp, 136]
	ldr	s0, [x0, 8]
	str	s0, [sp, 72]
	ldr	x0, [sp, 136]
	ldr	s0, [x0, 12]
	str	s0, [sp, 76]
	ldr	x0, [sp, 144]
	ldr	s0, [x0]
	str	s0, [sp, 80]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 4]
	str	s0, [sp, 84]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 8]
	str	s0, [sp, 88]
	ldr	x0, [sp, 144]
	ldr	s0, [x0, 12]
	str	s0, [sp, 92]
	ldr	x0, [sp, 152]
	ldr	s0, [x0]
	str	s0, [sp, 96]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 4]
	str	s0, [sp, 100]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 8]
	str	s0, [sp, 104]
	ldr	x0, [sp, 152]
	ldr	s0, [x0, 12]
	str	s0, [sp, 108]
	ldr	x0, [sp, 120]
	ldr	s0, [sp, 48]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 4
	ldr	s0, [sp, 52]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 8
	ldr	s0, [sp, 56]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 12
	ldr	s0, [sp, 60]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 16
	ldr	s0, [sp, 64]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 20
	ldr	s0, [sp, 68]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 24
	ldr	s0, [sp, 72]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 28
	ldr	s0, [sp, 76]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 32
	ldr	s0, [sp, 80]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 36
	ldr	s0, [sp, 84]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 40
	ldr	s0, [sp, 88]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 44
	ldr	s0, [sp, 92]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 48
	ldr	s0, [sp, 96]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 52
	ldr	s0, [sp, 100]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 56
	ldr	s0, [sp, 104]
	str	s0, [x0]
	ldr	x0, [sp, 120]
	add	x0, x0, 60
	ldr	s0, [sp, 108]
	str	s0, [x0]
	ldr	x0, [sp, 128]
	add	x0, x0, 16
	str	x0, [sp, 128]
	ldr	x0, [sp, 136]
	add	x0, x0, 16
	str	x0, [sp, 136]
	ldr	x0, [sp, 144]
	add	x0, x0, 16
	str	x0, [sp, 144]
	ldr	x0, [sp, 152]
	add	x0, x0, 16
	str	x0, [sp, 152]
	ldr	w0, [sp, 28]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 120]
	add	x0, x1, x0
	str	x0, [sp, 120]
	ldr	w0, [sp, 44]
	add	w0, w0, 4
	str	w0, [sp, 44]
	b	.L42
.L41:
	ldr	w0, [sp, 12]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 112]
	add	x0, x1, x0
	str	x0, [sp, 112]
	ldr	w0, [sp, 40]
	add	w0, w0, 4
	str	w0, [sp, 40]
	b	.L43
.L44:
	nop
	add	sp, sp, 160
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4103:
	.size	_Z5packBiiPfiS_, .-_Z5packBiiPfiS_
	.align	2
	.global	_Z10kernel_4x4iiiPfS_S_i
	.type	_Z10kernel_4x4iiiPfS_S_i, %function
_Z10kernel_4x4iiiPfS_S_i:
.LFB4104:
	.cfi_startproc
	sub	sp, sp, #1232
	.cfi_def_cfa_offset 1232
	stp	x29, x30, [sp]
	.cfi_offset 29, -1232
	.cfi_offset 30, -1224
	mov	x29, sp
	str	w0, [sp, 60]
	str	w1, [sp, 56]
	str	w2, [sp, 52]
	str	x3, [sp, 40]
	str	x4, [sp, 32]
	str	x5, [sp, 24]
	str	w6, [sp, 48]
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [x0]
	str	x1, [sp, 1224]
	mov	x1,0
	ldr	x0, [sp, 40]
	str	x0, [sp, 88]
	ldr	x0, [sp, 32]
	str	x0, [sp, 96]
	ldr	x0, [sp, 24]
	str	x0, [sp, 104]
	str	wzr, [sp, 76]
.L83:
	ldr	w0, [sp, 76]
	add	w0, w0, 3
	ldr	w1, [sp, 60]
	cmp	w1, w0
	ble	.L85
	str	wzr, [sp, 80]
.L82:
	ldr	w0, [sp, 80]
	add	w0, w0, 3
	ldr	w1, [sp, 56]
	cmp	w1, w0
	ble	.L47
	movi	v0.4s, 0
	str	q0, [sp, 256]
	movi	v0.4s, 0
	str	q0, [sp, 272]
	movi	v0.4s, 0
	str	q0, [sp, 288]
	movi	v0.4s, 0
	str	q0, [sp, 304]
	ldr	x0, [sp, 88]
	prfm	PLDL1KEEP, [x0]
	ldr	x0, [sp, 96]
	prfm	PLDL1KEEP, [x0]
	str	wzr, [sp, 84]
.L73:
	ldr	w0, [sp, 84]
	add	w0, w0, 3
	ldr	w1, [sp, 52]
	cmp	w1, w0
	ble	.L48
	ldr	x0, [sp, 88]
	str	x0, [sp, 168]
	ldr	x0, [sp, 168]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 320]
	ldr	x0, [sp, 96]
	str	x0, [sp, 160]
	ldr	x0, [sp, 160]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 336]
	ldr	q0, [sp, 256]
	str	q0, [sp, 992]
	ldr	q0, [sp, 336]
	str	q0, [sp, 1008]
	ldr	q0, [sp, 320]
	str	q0, [sp, 240]
	ldr	s0, [sp, 240]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 1008]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 992]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 256]
	ldr	q0, [sp, 272]
	str	q0, [sp, 960]
	ldr	q0, [sp, 336]
	str	q0, [sp, 976]
	ldr	q0, [sp, 320]
	str	q0, [sp, 240]
	ldr	s0, [sp, 244]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 976]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 960]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 272]
	ldr	q0, [sp, 288]
	str	q0, [sp, 928]
	ldr	q0, [sp, 336]
	str	q0, [sp, 944]
	ldr	q0, [sp, 320]
	str	q0, [sp, 240]
	ldr	s0, [sp, 248]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 944]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 928]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 288]
	ldr	q0, [sp, 304]
	str	q0, [sp, 896]
	ldr	q0, [sp, 336]
	str	q0, [sp, 912]
	ldr	q0, [sp, 320]
	str	q0, [sp, 240]
	ldr	s0, [sp, 252]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 912]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 896]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 304]
	ldr	x0, [sp, 88]
	add	x0, x0, 16
	str	x0, [sp, 152]
	ldr	x0, [sp, 152]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 352]
	ldr	x0, [sp, 96]
	add	x0, x0, 16
	str	x0, [sp, 144]
	ldr	x0, [sp, 144]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 368]
	ldr	q0, [sp, 256]
	str	q0, [sp, 864]
	ldr	q0, [sp, 368]
	str	q0, [sp, 880]
	ldr	q0, [sp, 352]
	str	q0, [sp, 240]
	ldr	s0, [sp, 240]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 880]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 864]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 256]
	ldr	q0, [sp, 272]
	str	q0, [sp, 832]
	ldr	q0, [sp, 368]
	str	q0, [sp, 848]
	ldr	q0, [sp, 352]
	str	q0, [sp, 240]
	ldr	s0, [sp, 244]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 848]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 832]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 272]
	ldr	q0, [sp, 288]
	str	q0, [sp, 800]
	ldr	q0, [sp, 368]
	str	q0, [sp, 816]
	ldr	q0, [sp, 352]
	str	q0, [sp, 240]
	ldr	s0, [sp, 248]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 816]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 800]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 288]
	ldr	q0, [sp, 304]
	str	q0, [sp, 768]
	ldr	q0, [sp, 368]
	str	q0, [sp, 784]
	ldr	q0, [sp, 352]
	str	q0, [sp, 240]
	ldr	s0, [sp, 252]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 784]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 768]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 304]
	ldr	x0, [sp, 88]
	add	x0, x0, 32
	str	x0, [sp, 136]
	ldr	x0, [sp, 136]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 384]
	ldr	x0, [sp, 96]
	add	x0, x0, 32
	str	x0, [sp, 128]
	ldr	x0, [sp, 128]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 400]
	ldr	q0, [sp, 256]
	str	q0, [sp, 736]
	ldr	q0, [sp, 400]
	str	q0, [sp, 752]
	ldr	q0, [sp, 384]
	str	q0, [sp, 240]
	ldr	s0, [sp, 240]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 752]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 736]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 256]
	ldr	q0, [sp, 272]
	str	q0, [sp, 704]
	ldr	q0, [sp, 400]
	str	q0, [sp, 720]
	ldr	q0, [sp, 384]
	str	q0, [sp, 240]
	ldr	s0, [sp, 244]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 720]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 704]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 272]
	ldr	q0, [sp, 288]
	str	q0, [sp, 672]
	ldr	q0, [sp, 400]
	str	q0, [sp, 688]
	ldr	q0, [sp, 384]
	str	q0, [sp, 240]
	ldr	s0, [sp, 248]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 688]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 672]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 288]
	ldr	q0, [sp, 304]
	str	q0, [sp, 640]
	ldr	q0, [sp, 400]
	str	q0, [sp, 656]
	ldr	q0, [sp, 384]
	str	q0, [sp, 240]
	ldr	s0, [sp, 252]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 656]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 640]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 304]
	ldr	x0, [sp, 88]
	add	x0, x0, 48
	str	x0, [sp, 120]
	ldr	x0, [sp, 120]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 416]
	ldr	x0, [sp, 96]
	add	x0, x0, 48
	str	x0, [sp, 112]
	ldr	x0, [sp, 112]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 432]
	ldr	q0, [sp, 256]
	str	q0, [sp, 608]
	ldr	q0, [sp, 432]
	str	q0, [sp, 624]
	ldr	q0, [sp, 416]
	str	q0, [sp, 240]
	ldr	s0, [sp, 240]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 624]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 608]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 256]
	ldr	q0, [sp, 272]
	str	q0, [sp, 576]
	ldr	q0, [sp, 432]
	str	q0, [sp, 592]
	ldr	q0, [sp, 416]
	str	q0, [sp, 240]
	ldr	s0, [sp, 244]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 592]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 576]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 272]
	ldr	q0, [sp, 288]
	str	q0, [sp, 544]
	ldr	q0, [sp, 432]
	str	q0, [sp, 560]
	ldr	q0, [sp, 416]
	str	q0, [sp, 240]
	ldr	s0, [sp, 248]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 560]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 544]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 288]
	ldr	q0, [sp, 304]
	str	q0, [sp, 512]
	ldr	q0, [sp, 432]
	str	q0, [sp, 528]
	ldr	q0, [sp, 416]
	str	q0, [sp, 240]
	ldr	s0, [sp, 252]
	dup	v0.4s, v0.s[0]
	mov	v1.16b, v0.16b
	ldr	q0, [sp, 528]
	fmul	v1.4s, v1.4s, v0.4s
	ldr	q0, [sp, 512]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 304]
	ldr	x0, [sp, 88]
	add	x0, x0, 64
	prfm	PLDL1KEEP, [x0]
	ldr	x0, [sp, 96]
	add	x0, x0, 64
	prfm	PLDL1KEEP, [x0]
	ldr	x0, [sp, 88]
	add	x0, x0, 64
	str	x0, [sp, 88]
	ldr	x0, [sp, 96]
	add	x0, x0, 64
	str	x0, [sp, 96]
	ldr	w0, [sp, 84]
	add	w0, w0, 4
	str	w0, [sp, 84]
	b	.L73
.L48:
	ldr	w0, [sp, 52]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	neg	x0, x0
	ldr	x1, [sp, 88]
	add	x0, x1, x0
	str	x0, [sp, 88]
	ldr	x0, [sp, 104]
	str	x0, [sp, 232]
	ldr	x0, [sp, 232]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 448]
	ldrsw	x0, [sp, 48]
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 224]
	ldr	x0, [sp, 224]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 464]
	ldr	w0, [sp, 48]
	lsl	w0, w0, 1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 216]
	ldr	x0, [sp, 216]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 480]
	ldr	w1, [sp, 48]
	mov	w0, w1
	lsl	w0, w0, 1
	add	w0, w0, w1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 208]
	ldr	x0, [sp, 208]
	ldr	q0, [x0]
	nop
	str	q0, [sp, 496]
	ldr	q0, [sp, 448]
	str	q0, [sp, 1184]
	ldr	q0, [sp, 256]
	str	q0, [sp, 1200]
	ldr	q1, [sp, 1184]
	ldr	q0, [sp, 1200]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 448]
	ldr	q0, [sp, 464]
	str	q0, [sp, 1152]
	ldr	q0, [sp, 272]
	str	q0, [sp, 1168]
	ldr	q1, [sp, 1152]
	ldr	q0, [sp, 1168]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 464]
	ldr	q0, [sp, 480]
	str	q0, [sp, 1120]
	ldr	q0, [sp, 288]
	str	q0, [sp, 1136]
	ldr	q1, [sp, 1120]
	ldr	q0, [sp, 1136]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 480]
	ldr	q0, [sp, 496]
	str	q0, [sp, 1088]
	ldr	q0, [sp, 304]
	str	q0, [sp, 1104]
	ldr	q1, [sp, 1088]
	ldr	q0, [sp, 1104]
	fadd	v0.4s, v1.4s, v0.4s
	str	q0, [sp, 496]
	ldr	x0, [sp, 104]
	str	x0, [sp, 200]
	ldr	q0, [sp, 448]
	str	q0, [sp, 1072]
	ldr	x0, [sp, 200]
	ldr	q0, [sp, 1072]
	str	q0, [x0]
	nop
	ldrsw	x0, [sp, 48]
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 192]
	ldr	q0, [sp, 464]
	str	q0, [sp, 1056]
	ldr	x0, [sp, 192]
	ldr	q0, [sp, 1056]
	str	q0, [x0]
	nop
	ldr	w0, [sp, 48]
	lsl	w0, w0, 1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 184]
	ldr	q0, [sp, 480]
	str	q0, [sp, 1040]
	ldr	x0, [sp, 184]
	ldr	q0, [sp, 1040]
	str	q0, [x0]
	nop
	ldr	w1, [sp, 48]
	mov	w0, w1
	lsl	w0, w0, 1
	add	w0, w0, w1
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 104]
	add	x0, x1, x0
	str	x0, [sp, 176]
	ldr	q0, [sp, 496]
	str	q0, [sp, 1024]
	ldr	x0, [sp, 176]
	ldr	q0, [sp, 1024]
	str	q0, [x0]
	nop
	ldr	x0, [sp, 104]
	add	x0, x0, 16
	str	x0, [sp, 104]
	ldr	w0, [sp, 80]
	add	w0, w0, 4
	str	w0, [sp, 80]
	b	.L82
.L47:
	ldr	x0, [sp, 32]
	str	x0, [sp, 96]
	ldr	w0, [sp, 52]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 88]
	add	x0, x1, x0
	str	x0, [sp, 88]
	ldr	w0, [sp, 48]
	lsl	w0, w0, 2
	sxtw	x0, w0
	lsl	x0, x0, 2
	ldr	x1, [sp, 24]
	add	x0, x1, x0
	str	x0, [sp, 24]
	ldr	x0, [sp, 24]
	str	x0, [sp, 104]
	ldr	w0, [sp, 76]
	add	w0, w0, 4
	str	w0, [sp, 76]
	b	.L83
.L85:
	nop
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x1, [sp, 1224]
	ldr	x2, [x0]
	subs	x1, x1, x2
	mov	x2, 0
	beq	.L84
	bl	__stack_chk_fail
.L84:
	ldp	x29, x30, [sp]
	add	sp, sp, 1232
	.cfi_restore 29
	.cfi_restore 30
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE4104:
	.size	_Z10kernel_4x4iiiPfS_S_i, .-_Z10kernel_4x4iiiPfS_S_i
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
