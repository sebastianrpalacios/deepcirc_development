module m0x2A (input in2, in1, in3, output out);

	wire \$n5_0;

	nor (\$n5_0, in2, in1);
	nor (out, \$n5_0, in3);

endmodule
