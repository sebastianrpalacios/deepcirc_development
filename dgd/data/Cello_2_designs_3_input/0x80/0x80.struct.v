module m0x80 (input in2, in1, in3, output out);

	wire \$n6_0;
	wire \$n5_0;

	nor (\$n5_0, in2, in3);
	not (\$n6_0, \$n5_0);
	nor (out, in1, \$n6_0);

endmodule
