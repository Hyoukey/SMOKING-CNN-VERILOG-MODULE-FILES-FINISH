module layer1 (
    input clk,
    input rst_n,
    input [7:0] pixel_in,
    input valid_in,
    // Outputs: 3 Filters, 12-bit signed data (after pooling)
    output reg signed [11:0] out0,
    output reg signed [11:0] out1,
    output reg signed [11:0] out2,
    output reg valid_out
);

    // --- 1. WEIGHTS (3 Filters x 9 Weights = 27 Total) ---
    wire signed [7:0] weight [0:26];

    // ------------------------------------------------------------------------
    // PASTE YOUR 'rom_conv1.txt' CONTENT BELOW
    // ------------------------------------------------------------------------
    
    // [PASTE HERE]
assign weight[0] = 8'h65;
assign weight[1] = 8'h09;
assign weight[2] = 8'h43;
assign weight[3] = 8'hd2;
assign weight[4] = 8'h0c;
assign weight[5] = 8'h32;
assign weight[6] = 8'h95;
assign weight[7] = 8'he6;
assign weight[8] = 8'hf0;
assign weight[9] = 8'heb;
assign weight[10] = 8'hfd;
assign weight[11] = 8'h1b;
assign weight[12] = 8'hec;
assign weight[13] = 8'hf6;
assign weight[14] = 8'h15;
assign weight[15] = 8'hb4;
assign weight[16] = 8'hae;
assign weight[17] = 8'h81;
assign weight[18] = 8'h70;
assign weight[19] = 8'he8;
assign weight[20] = 8'hfe;
assign weight[21] = 8'h4e;
assign weight[22] = 8'h29;
assign weight[23] = 8'hd1;
assign weight[24] = 8'h1f;
assign weight[25] = 8'h28;
assign weight[26] = 8'h09;
    
    // ------------------------------------------------------------------------


    // --- 2. LINE BUFFER (28x28 Input, 8-bit) ---
    wire [7:0] w00, w01, w02, w10, w11, w12, w20, w21, w22;
    wire win_valid;

    conv_buf #(.WIDTH(28), .DATA_BITS(8)) lb (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(pixel_in),
        .valid_in(valid_in),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22),
        .valid_out(win_valid)
    );

    // --- 3. CONVOLUTION (3 Parallel Filters) ---
    // Note: Outputs are now 32-bit to prevent overflow before ReLU
    wire signed [31:0] raw_conv0, raw_conv1, raw_conv2;
    wire conv_valid; 

    // Filter 0 (Weights 0-8)
    conv_calc #(.DATA_BITS(8)) pe0 (
        .clk(clk), .rst_n(rst_n), .valid_in(win_valid),
        .p00(w00), .p01(w01), .p02(w02), .p10(w10), .p11(w11), .p12(w12), .p20(w20), .p21(w21), .p22(w22),
        .k00(weight[0]), .k01(weight[1]), .k02(weight[2]),
        .k10(weight[3]), .k11(weight[4]), .k12(weight[5]),
        .k20(weight[6]), .k21(weight[7]), .k22(weight[8]),
        .result(raw_conv0), .valid_out(conv_valid)
    );

    // Filter 1 (Weights 9-17)
    conv_calc #(.DATA_BITS(8)) pe1 (
        .clk(clk), .rst_n(rst_n), .valid_in(win_valid),
        .p00(w00), .p01(w01), .p02(w02), .p10(w10), .p11(w11), .p12(w12), .p20(w20), .p21(w21), .p22(w22),
        .k00(weight[9]),  .k01(weight[10]), .k02(weight[11]),
        .k10(weight[12]), .k11(weight[13]), .k12(weight[14]),
        .k20(weight[15]), .k21(weight[16]), .k22(weight[17]),
        .result(raw_conv1), .valid_out()
    );

    // Filter 2 (Weights 18-26)
    conv_calc #(.DATA_BITS(8)) pe2 (
        .clk(clk), .rst_n(rst_n), .valid_in(win_valid),
        .p00(w00), .p01(w01), .p02(w02), .p10(w10), .p11(w11), .p12(w12), .p20(w20), .p21(w21), .p22(w22),
        .k00(weight[18]), .k01(weight[19]), .k02(weight[20]),
        .k10(weight[21]), .k11(weight[22]), .k12(weight[23]),
        .k20(weight[24]), .k21(weight[25]), .k22(weight[26]),
        .result(raw_conv2), .valid_out()
    );

    // --- 4. POOLING & ACTIVATION (ReLU + Scaling) ---
    // Output Size: 26x26 -> 13x13
    
    reg [9:0] x_cnt, y_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= 0; y_cnt <= 0;
            valid_out <= 0;
            out0 <= 0; out1 <= 0; out2 <= 0;
        end else if (conv_valid) begin
            
            // Subsampling Logic (Take Top-Left of 2x2)
            if (x_cnt[0] == 0 && y_cnt[0] == 0) begin
                valid_out <= 1;
                
                // APPLY ReLU and SCALE (Divide by 16 by taking bits [15:4])
                // Check if negative -> 0. Else -> Scale.
                out0 <= (raw_conv0 > 0) ? raw_conv0[15:4] : 12'd0;
                out1 <= (raw_conv1 > 0) ? raw_conv1[15:4] : 12'd0;
                out2 <= (raw_conv2 > 0) ? raw_conv2[15:4] : 12'd0;
                
            end else begin
                valid_out <= 0;
            end

            // Coordinate Counters (0 to 25)
            // The Conv output is 26x26 pixels (28 - 3 + 1)
            if (x_cnt == 25) begin
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