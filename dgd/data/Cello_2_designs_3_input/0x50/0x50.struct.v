module m0x50 (input in2, in1, in3, output out);

	wire \$n5_0;

	not (\$n5_0, in3);
	nor (out, in1, \$n5_0);

endmodule
