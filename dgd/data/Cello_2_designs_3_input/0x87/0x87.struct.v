module m0x87 (input in2, in1, in3, output out);

	wire \$n10_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n6_1;
	wire \$n5_0;

	not (\$n5_0, in1);
	nor (\$n6_0, in2, in3);
	nor (\$n6_1, in2, in3);
	not (\$n7_0, \$n6_1);
	nor (\$n8_0, \$n7_0, in1);
	nor (\$n9_0, \$n6_0, \$n5_0);
	nor (\$n10_0, \$n9_0, \$n8_0);
	not (out, \$n10_0);

endmodule
