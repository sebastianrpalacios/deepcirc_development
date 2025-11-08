module m0x08 (input in2, in1, in3, output out);

	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	nor (\$n6_0, in2, in3);
	not (\$n5_0, in1);
	not (\$n7_0, \$n6_0);
	nor (out, \$n7_0, \$n5_0);

endmodule
