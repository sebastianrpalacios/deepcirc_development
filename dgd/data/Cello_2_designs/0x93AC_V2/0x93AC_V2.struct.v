module m0x93AC (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n11_0;
	wire \$n11_1;
	wire \$n14_0;
	wire \$n15_0;
	wire \$n9_0;
	wire \$n9_1;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n13_0;
	wire \$n7_0;
	wire \$n18_0;
	wire \$n19_0;
	wire \$n16_0;
	wire \$n17_0;
	wire \$n6_0;

	not (\$n12_0, \$n11_0);
	not (\$n10_0, \$n9_0);
	not (\$n7_0, in3);
	nor (\$n9_0, in2, in4);
	nor (\$n9_1, in2, in4);
	nor (\$n11_0, \$n7_0, in1);
	nor (\$n11_1, \$n7_0, in1);
	nor (\$n17_0, \$n11_1, \$n9_1);
	not (\$n6_0, in2);
	nor (\$n14_0, in3, \$n6_0);
	not (\$n8_0, in1);
	not (\$n15_0, \$n14_0);
	nor (\$n16_0, \$n15_0, \$n8_0);
	not (\$n18_0, \$n17_0);
	nor (\$n19_0, \$n18_0, \$n16_0);
	nor (\$n13_0, \$n10_0, \$n12_0);
	nor (out, \$n13_0, \$n19_0);

endmodule
