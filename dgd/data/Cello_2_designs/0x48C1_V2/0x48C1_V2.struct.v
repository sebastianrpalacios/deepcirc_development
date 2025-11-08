module m0x48C1 (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n11_0;
	wire \$n14_0;
	wire \$n15_0;
	wire \$n9_0;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n13_0;
	wire \$n13_1;
	wire \$n7_0;
	wire \$n16_0;
	wire \$n17_0;
	wire \$n6_0;

	not (\$n6_0, in4);
	not (\$n9_0, in1);
	not (\$n8_0, in3);
	nor (\$n10_0, \$n9_0, \$n6_0);
	not (\$n14_0, \$n13_0);
	nor (\$n12_0, in1, in4);
	not (\$n7_0, in2);
	nor (\$n13_0, \$n12_0, in3);
	nor (\$n13_1, \$n12_0, in3);
	nor (\$n16_0, \$n13_1, \$n7_0);
	nor (\$n15_0, \$n14_0, in2);
	nor (\$n17_0, \$n15_0, \$n16_0);
	nor (\$n11_0, \$n10_0, \$n8_0);
	nor (out, \$n11_0, \$n17_0);

endmodule
