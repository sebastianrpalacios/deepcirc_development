module m0x0A (input in2, in1, in3, output out);

	wire \$n5_0;

	not (\$n5_0, in1);
	nor (out, \$n5_0, in3);

endmodule
