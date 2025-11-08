module m0x0575 (input in2, in1, in4, in3, output out);

	wire \$n6_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;

	not (\$n6_0, in3);
	nor (\$n7_0, \$n6_0, in2);
	nor (\$n8_0, \$n7_0, in4);
	nor (\$n9_0, in2, in1);
	nor (out, \$n9_0, \$n8_0);

endmodule
