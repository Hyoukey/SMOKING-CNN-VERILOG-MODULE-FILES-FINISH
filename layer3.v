module layer3 (
    input clk,
    input rst_n,
    // 3 Input Channels (12-bit signed data from Layer 2)
    input signed [11:0] in0, in1, in2,
    input valid_in,
    // Outputs: 3 Filters, 12-bit signed data (1x1 Pixel each)
    output reg signed [11:0] out0, out1, out2,
    output reg valid_out
);

    // --- 1. WEIGHTS WIRE (3 Filters x 3 Channels x 9 Kernel = 81 Weights) ---
    wire signed [7:0] weight [0:80];

    // ------------------------------------------------------------------------
    // PASTE YOUR 'rom_conv3.txt' CONTENT BELOW
    // ------------------------------------------------------------------------
    
    // [PASTE HERE]
assign weight[0] = 8'h37;
assign weight[1] = 8'h06;
assign weight[2] = 8'h18;
assign weight[3] = 8'h17;
assign weight[4] = 8'h12;
assign weight[5] = 8'he5;
assign weight[6] = 8'hff;
assign weight[7] = 8'h4b;
assign weight[8] = 8'h74;
assign weight[9] = 8'hf5;
assign weight[10] = 8'hfe;
assign weight[11] = 8'h06;
assign weight[12] = 8'h07;
assign weight[13] = 8'he5;
assign weight[14] = 8'hd2;
assign weight[15] = 8'h16;
assign weight[16] = 8'h38;
assign weight[17] = 8'h20;
assign weight[18] = 8'h02;
assign weight[19] = 8'h71;
assign weight[20] = 8'h31;
assign weight[21] = 8'h6c;
assign weight[22] = 8'h55;
assign weight[23] = 8'h28;
assign weight[24] = 8'hfb;
assign weight[25] = 8'h12;
assign weight[26] = 8'hf3;
assign weight[27] = 8'hf5;
assign weight[28] = 8'h31;
assign weight[29] = 8'hf3;
assign weight[30] = 8'h3a;
assign weight[31] = 8'h0e;
assign weight[32] = 8'hdb;
assign weight[33] = 8'h03;
assign weight[34] = 8'hf4;
assign weight[35] = 8'hd5;
assign weight[36] = 8'h55;
assign weight[37] = 8'hd8;
assign weight[38] = 8'h2f;
assign weight[39] = 8'h33;
assign weight[40] = 8'hff;
assign weight[41] = 8'h0c;
assign weight[42] = 8'h0b;
assign weight[43] = 8'hda;
assign weight[44] = 8'h00;
assign weight[45] = 8'h0c;
assign weight[46] = 8'hc8;
assign weight[47] = 8'hfd;
assign weight[48] = 8'h02;
assign weight[49] = 8'hf5;
assign weight[50] = 8'hb0;
assign weight[51] = 8'h0a;
assign weight[52] = 8'h26;
assign weight[53] = 8'h65;
assign weight[54] = 8'h06;
assign weight[55] = 8'h46;
assign weight[56] = 8'hf4;
assign weight[57] = 8'hff;
assign weight[58] = 8'h07;
assign weight[59] = 8'h08;
assign weight[60] = 8'hdf;
assign weight[61] = 8'h1e;
assign weight[62] = 8'h81;
assign weight[63] = 8'h23;
assign weight[64] = 8'h06;
assign weight[65] = 8'h60;
assign weight[66] = 8'hc5;
assign weight[67] = 8'hce;
assign weight[68] = 8'h16;
assign weight[69] = 8'h9b;
assign weight[70] = 8'hb5;
assign weight[71] = 8'hfd;
assign weight[72] = 8'h1d;
assign weight[73] = 8'h4b;
assign weight[74] = 8'hcc;
assign weight[75] = 8'hf0;
assign weight[76] = 8'he1;
assign weight[77] = 8'hda;
assign weight[78] = 8'h4e;
assign weight[79] = 8'h20;
assign weight[80] = 8'h2b;
    
    // ------------------------------------------------------------------------

    // --- 2. LINE BUFFERS (5x5 Input, 12-bit) ---
    wire [11:0] w0_00, w0_01, w0_02, w0_10, w0_11, w0_12, w0_20, w0_21, w0_22;
    wire [11:0] w1_00, w1_01, w1_02, w1_10, w1_11, w1_12, w1_20, w1_21, w1_22;
    wire [11:0] w2_00, w2_01, w2_02, w2_10, w2_11, w2_12, w2_20, w2_21, w2_22;
    wire win_valid;

    // Buffer 0 (Channel 0)
    conv_buf #(.WIDTH(5), .DATA_BITS(12)) lb0 (
        .clk(clk), .rst_n(rst_n), .data_in(in0), .valid_in(valid_in),
        .w00(w0_00), .w01(w0_01), .w02(w0_02), .w10(w0_10), .w11(w0_11), .w12(w0_12), .w20(w0_20), .w21(w0_21), .w22(w0_22),
        .valid_out(win_valid)
    );
    // Buffer 1 (Channel 1)
    conv_buf #(.WIDTH(5), .DATA_BITS(12)) lb1 (
        .clk(clk), .rst_n(rst_n), .data_in(in1), .valid_in(valid_in),
        .w00(w1_00), .w01(w1_01), .w02(w1_02), .w10(w1_10), .w11(w1_11), .w12(w1_12), .w20(w1_20), .w21(w1_21), .w22(w1_22),
        .valid_out()
    );
    // Buffer 2 (Channel 2)
    conv_buf #(.WIDTH(5), .DATA_BITS(12)) lb2 (
        .clk(clk), .rst_n(rst_n), .data_in(in2), .valid_in(valid_in),
        .w00(w2_00), .w01(w2_01), .w02(w2_02), .w10(w2_10), .w11(w2_11), .w12(w2_12), .w20(w2_20), .w21(w2_21), .w22(w2_22),
        .valid_out()
    );

    // --- 3. CONVOLUTION (9 Calculators) ---
    wire signed [31:0] f0_c0, f0_c1, f0_c2;
    wire signed [31:0] f1_c0, f1_c1, f1_c2;
    wire signed [31:0] f2_c0, f2_c1, f2_c2;
    wire calc_valid;

    // Filter 0
    conv_calc #(.DATA_BITS(12)) c0_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[0]), .k01(weight[1]), .k02(weight[2]), .k10(weight[3]), .k11(weight[4]), .k12(weight[5]), .k20(weight[6]), .k21(weight[7]), .k22(weight[8]), .result(f0_c0), .valid_out(calc_valid));
    conv_calc #(.DATA_BITS(12)) c0_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[9]), .k01(weight[10]), .k02(weight[11]), .k10(weight[12]), .k11(weight[13]), .k12(weight[14]), .k20(weight[15]), .k21(weight[16]), .k22(weight[17]), .result(f0_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c0_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[18]), .k01(weight[19]), .k02(weight[20]), .k10(weight[21]), .k11(weight[22]), .k12(weight[23]), .k20(weight[24]), .k21(weight[25]), .k22(weight[26]), .result(f0_c2), .valid_out());

    // Filter 1
    conv_calc #(.DATA_BITS(12)) c1_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[27]), .k01(weight[28]), .k02(weight[29]), .k10(weight[30]), .k11(weight[31]), .k12(weight[32]), .k20(weight[33]), .k21(weight[34]), .k22(weight[35]), .result(f1_c0), .valid_out());
    conv_calc #(.DATA_BITS(12)) c1_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[36]), .k01(weight[37]), .k02(weight[38]), .k10(weight[39]), .k11(weight[40]), .k12(weight[41]), .k20(weight[42]), .k21(weight[43]), .k22(weight[44]), .result(f1_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c1_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[45]), .k01(weight[46]), .k02(weight[47]), .k10(weight[48]), .k11(weight[49]), .k12(weight[50]), .k20(weight[51]), .k21(weight[52]), .k22(weight[53]), .result(f1_c2), .valid_out());

    // Filter 2
    conv_calc #(.DATA_BITS(12)) c2_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[54]), .k01(weight[55]), .k02(weight[56]), .k10(weight[57]), .k11(weight[58]), .k12(weight[59]), .k20(weight[60]), .k21(weight[61]), .k22(weight[62]), .result(f2_c0), .valid_out());
    conv_calc #(.DATA_BITS(12)) c2_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[63]), .k01(weight[64]), .k02(weight[65]), .k10(weight[66]), .k11(weight[67]), .k12(weight[68]), .k20(weight[69]), .k21(weight[70]), .k22(weight[71]), .result(f2_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c2_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[72]), .k01(weight[73]), .k02(weight[74]), .k10(weight[75]), .k11(weight[76]), .k12(weight[77]), .k20(weight[78]), .k21(weight[79]), .k22(weight[80]), .result(f2_c2), .valid_out());

    // --- 4. SUMMATION + POOLING (Max of Top-Left 2x2) ---
    // Conv Output size is 3x3.
    // MaxPool(2) on 3x3 only looks at (0,0), (0,1), (1,0), (1,1).
    // Result is a single value (1x1).
    
    reg [9:0] x_cnt, y_cnt;
    reg signed [31:0] sum0, sum1, sum2;
    
    // Registers to hold the max value found so far
    reg signed [31:0] max0, max1, max2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            x_cnt <= 0; y_cnt <= 0;
            out0 <= 0; out1 <= 0; out2 <= 0;
            max0 <= 0; max1 <= 0; max2 <= 0;
        end else if (calc_valid) begin
            
            // A. Sum Channels
            sum0 = f0_c0 + f0_c1 + f0_c2;
            sum1 = f1_c0 + f1_c1 + f1_c2;
            sum2 = f2_c0 + f2_c1 + f2_c2;

            // B. Pooling Logic (Find Max of the 2x2 block: 0,0 0,1 1,0 1,1)
            
            // If we are at (0,0), initialize the max register
            if (x_cnt == 0 && y_cnt == 0) begin
                max0 <= sum0;
                max1 <= sum1;
                max2 <= sum2;
            end 
            // If we are in the 2x2 zone (x<2, y<2), compare and update
            else if (x_cnt < 2 && y_cnt < 2) begin
                if (sum0 > max0) max0 <= sum0;
                if (sum1 > max1) max1 <= sum1;
                if (sum2 > max2) max2 <= sum2;
            end

            // C. Output Logic
            // When we hit (1,1), we have seen all 4 pixels of the 2x2 block.
            // Output the final max value.
            if (x_cnt == 1 && y_cnt == 1) begin
                valid_out <= 1;
                // ReLU + Scale (Divide by 128 = >> 7)
                out0 <= (max0 > 0) ? max0[18:7] : 12'd0;
                out1 <= (max1 > 0) ? max1[18:7] : 12'd0;
                out2 <= (max2 > 0) ? max2[18:7] : 12'd0;
            end else begin
                valid_out <= 0;
            end

            // Coordinate Counters (0 to 2 for 3x3 image)
            if (x_cnt == 2) begin
                x_cnt <= 0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end else begin
            valid_out <= 0;
        end
    end

endmodule