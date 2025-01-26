#include <random>

#include <spdlog/spdlog.h>

#include "Labs/1-Drawing2D/tasks.h"

#include<algorithm>
#include<cmath>
#include<cstring>
using VCX::Labs::Common::ImageRGB;

namespace VCX::Labs::Drawing2D {
    /******************* 1.Image Dithering *****************/
    void DitheringThreshold(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }
    ImageRGB my_copy_img(ImageRGB const & input) {
        ImageRGB copy_img(input.GetSizeX(), input.GetSizeY());
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                copy_img.At(x, y) = input.At(x, y);
            }
        return copy_img;
    }
    void DitheringRandomUniform(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        std::random_device my_rd;        
        std::mt19937 my_gen(my_rd()); 
        std::uniform_real_distribution<float> dis(-0.5, 0.5); 
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                float my_rand_noise = dis(my_gen);
                output.At(x, y) = {
                    color.r+my_rand_noise > 0.5 ? 1 : 0,
                    color.g+my_rand_noise > 0.5 ? 1 : 0,
                    color.b+my_rand_noise > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringRandomBlueNoise(
        ImageRGB &       output,
        ImageRGB const & input,
        ImageRGB const & noise) {
        // your code here:
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                glm::vec3 noise_color   = noise.At(x, y);
                float     blue_noise  = noise_color.r;
                output.At(x, y)         = {
                    color.r + blue_noise > 1 ? 1 : 0,
                    color.g + blue_noise > 1 ? 1 : 0,
                    color.b + blue_noise > 1 ? 1 : 0,
                };
            }
    }

    void DitheringOrdered(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        int shake_lst[9] = { 6, 8, 4, 1, 0, 3, 5, 2, 7 };
        for (int i = 0; i < input.GetSizeX(); i++)
        {
            for (int j = 0; j < input.GetSizeY(); j++)
            {
                for (int r_x = 0; r_x < 3; r_x++)
                {
                    for (int r_y = 0; r_y < 3; r_y++)
                    {
                        int x_index = i * 3 + r_x;
                        int y_index = j * 3 + r_y;
                        float check_ = (glm::vec3(input.At(i, j))).r * 9;
                        output.At(x_index, y_index) = (check_ > (float) shake_lst[r_x * 3 + r_y]) ? glm::vec3({ 1, 1, 1 }) : glm::vec3({ 0, 0, 0 });
                    }
                }
            }
        }
    }

    void DitheringErrorDiffuse(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:

        float out_record[10000][2];
        memset(out_record, 0, sizeof(out_record));

        int line_index = 0;
        for (int j = 0; j < input.GetSizeY(); j++) {
            for (int i = 0; i < input.GetSizeX(); i++) {
            
                float output_former = out_record[i][line_index];
                out_record[i][line_index] = 0;

                float dither_ans = ((glm::vec3(input.At(i, j)).r+output_former) > 0.5) ? 1.0 : 0.0;
                output.At(i, j)    = glm::vec3({ dither_ans, dither_ans, dither_ans });

                float dither_noise = glm::vec3(input.At(i, j)).r+output_former - dither_ans;

                if (i+1<input.GetSizeX()) {
                    float tmp_          = dither_noise * (float) 7 / (float) 16;
                    out_record[i + 1][line_index] += tmp_;

                    if (j + 1 < input.GetSizeY())
                    {
                        float tmp_2             =  dither_noise * (float) 1 / (float) 16;
                        out_record[i][1 - line_index] += tmp_2;
                    }
                 }

                if (j + 1 < input.GetSizeY())
                {
                     float tmp_          =  dither_noise * (float) 5 / (float) 16;
                    out_record[i][1 - line_index]+=tmp_;
                     if (i != 0)
                     {
                         float tmp_2             =  dither_noise * (float) 3 / (float) 16;
                         out_record[i - 1][1 - line_index] += tmp_2;
                     }
                }

            }
            line_index = 1 - line_index;
        }
    }

    /******************* 2.Image Filtering *****************/
    void Blur(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        for (int i = 0; i < output.GetSizeX(); i++)
        {
            for (int j = 0; j < output.GetSizeY(); j++)
            {
                float tmp_cnt = 0;
                float tar_pixel_color_r = 0;
                float tar_pixel_color_g = 0;
                float tar_pixel_color_b = 0;
                for (int i_x = -1; i_x <= 1; i_x++)
                {
                    for (int j_y = -1; j_y <= 1; j_y++)
                    {
                        int tar_x = i + i_x;
                        int tar_y = j + j_y;
                        if (tar_x != -1 && tar_x != output.GetSizeX() && tar_y != -1 && tar_y != output.GetSizeY())
                        {
                            tmp_cnt++;
                            tar_pixel_color_r += (input.At(tar_x, tar_y).r);
                            tar_pixel_color_g += (input.At(tar_x, tar_y).g);
                            tar_pixel_color_b += (input.At(tar_x, tar_y).b);
                            /*printf("%f %f %f \n", tar_pixel_color_r, tar_pixel_color_g, tar_pixel_color_b);*/

                        }

                    }
                }
                //printf("%f %f %f cnt:%f\n", tar_pixel_color_r, tar_pixel_color_g, tar_pixel_color_b,tmp_cnt);
                output.At(i, j)   = { tar_pixel_color_r / tmp_cnt, tar_pixel_color_g / tmp_cnt, tar_pixel_color_b / tmp_cnt };
            }
        }
    }

    void Edge(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        float x_kernel[9] = {-1, 0, 1,-2, 0, 2, -1,  0,  1 };
        float y_kernel[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
        for (int i = 0; i < output.GetSizeX(); i++) {
            for (int j = 0; j < output.GetSizeY(); j++) {
                float tmp_cntx           = 0;
                float tar_pixel_color_rx = 0;
                float tar_pixel_color_gx = 0;
                float tar_pixel_color_bx = 0;

                float tmp_cnty           = 0;
                float tar_pixel_color_ry = 0;
                float tar_pixel_color_gy = 0;
                float tar_pixel_color_by = 0;

                for (int i_x = -1; i_x <= 1; i_x++) {
                    for (int j_y = -1; j_y <= 1; j_y++) {
                        int tar_x = i + i_x;
                        int tar_y = j + j_y;
                        int tar_index = (j_y + 1) * 3 + (i_x + 1);
                        if (tar_x != -1 && tar_x != output.GetSizeX() && tar_y != -1 && tar_y != output.GetSizeY()) {
                            tmp_cntx++;
                            tar_pixel_color_rx += (input.At(tar_x, tar_y).r)*x_kernel[tar_index];
                            tar_pixel_color_gx += (input.At(tar_x, tar_y).g) * x_kernel[tar_index];
                            tar_pixel_color_bx += (input.At(tar_x, tar_y).b) * x_kernel[tar_index];
                            /*printf("%f %f %f \n", tar_pixel_color_r, tar_pixel_color_g, tar_pixel_color_b);*/
                            tmp_cnty++;
                            tar_pixel_color_ry += (input.At(tar_x, tar_y).r) * y_kernel[tar_index];
                            tar_pixel_color_gy += (input.At(tar_x, tar_y).g) * y_kernel[tar_index];
                            tar_pixel_color_by += (input.At(tar_x, tar_y).b) * y_kernel[tar_index];
                        }
                    }
                }
                // printf("%f %f %f cnt:%f\n", tar_pixel_color_r, tar_pixel_color_g, tar_pixel_color_b,tmp_cnt);
                output.At(i, j) = { sqrt(pow(tar_pixel_color_rx, 2) + pow(tar_pixel_color_ry, 2)), sqrt(pow(tar_pixel_color_gx, 2) + pow(tar_pixel_color_gy, 2)),sqrt(pow(tar_pixel_color_bx, 2) + pow(tar_pixel_color_by, 2)) };
            }
        }
    }

    /******************* 3. Image Inpainting *****************/
    void Inpainting(
        ImageRGB &         output,
        ImageRGB const &   inputBack,
        ImageRGB const &   inputFront,
        const glm::ivec2 & offset) {
        output             = inputBack;
        std::size_t width  = inputFront.GetSizeX();
        std::size_t height = inputFront.GetSizeY();
        glm::vec3 * g      = new glm::vec3[width * height];
        memset(g, 0, sizeof(glm::vec3) * width * height);
        // set boundary condition
        for (std::size_t y = 0; y < height; ++y) {
            // set boundary for (0, y), your code: g[y * width] = ?
            g[y * width] = glm::vec3(inputFront.At(0, y)) * glm::vec3(-1) + glm::vec3(inputBack.At(offset.x, offset.y+y));

            // set boundary for (width - 1, y), your code: g[y * width + width - 1] = ?
            g[y * width + width - 1] = glm::vec3(inputFront.At(width - 1, y)) * glm::vec3(-1) + glm::vec3(inputBack.At(offset.x + width - 1, y + offset.y));
        }
        for (std::size_t x = 0; x < width; ++x) {

            // set boundary for (x, 0), your code: g[x] = ?
            g[x] = glm::vec3(inputFront.At(x, 0)) * glm::vec3(-1) + glm::vec3(inputBack.At(offset.x+x, offset.y));
            // set boundary for (x, height - 1), your code: g[(height - 1) * width + x] = ?
            g[x + (height - 1) * width] = glm::vec3(inputFront.At(x, height - 1)) * glm::vec3(-1) + glm::vec3(inputBack.At(offset.x+x, offset.y + height - 1));
        }


        // Jacobi iteration, solve Ag = b
        for (int iter = 0; iter < 8000; ++iter) {
            for (std::size_t y = 1; y < height - 1; ++y)
                for (std::size_t x = 1; x < width - 1; ++x) {
                    g[y * width + x] = (g[(y - 1) * width + x] + g[(y + 1) * width + x] + g[y * width + x - 1] + g[y * width + x + 1]);
                    g[y * width + x] = g[y * width + x] * glm::vec3(0.25);
                }
        }

        for (std::size_t y = 0; y < inputFront.GetSizeY(); ++y)
            for (std::size_t x = 0; x < inputFront.GetSizeX(); ++x) {
                glm::vec3 color = g[y * width + x] + inputFront.At(x, y);
                output.At(x + offset.x, y + offset.y) = color;
            }
        delete[] g;
    }

    /******************* 4. Line Drawing *****************/
    void DrawLine(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1) {
        // your code here:
        if (p0.x == p1.x)
        {
            for (int i = p0.y; i <= p1.y; i++) canvas.At(p0.x, i) = color;
        }
        else if (p0.y == p1.y)
        {
            for (int i = p0.x; i <= p1.x; i++) canvas.At(i,p0.y) = color;
        }
        else
        {
            glm::ivec2 p_start=p0, p_finish=p1;
            if (p0.x > p1.x)
            {
                p_start = p1;
                p_finish = p0;
            }

            double x_start = p_start.x;
            double x_finish = p_finish.x;
            double y_start = p_start.y;
            double y_finish = p_finish.y;
            float k = (y_finish - y_start) / (x_finish - x_start);
            int   dx       = (x_finish - x_start)*2;
            int   dy       = (y_finish - y_start)*2;

            if (abs(k) <= 1)
            {
                if (k > 0)
                {
                    int dydx = dy - dx, F = dy - dx / 2;
                    int y = y_start;
                    for (int x = x_start; x <= x_finish; x++)
                    {
                        canvas.At(x, y) = color;
                        if (F < 0) F += dy;
                        else {
                            y++;
                            F += dydx;
                        }
                    }
                }
                else
                {
                    //printf("haha1 %f\n",k);
                    int dydx = dy + dx, F = dy + dx / 2;
                    int y = y_start;
                    for (int x = x_start; x <= x_finish; x++) {
                        canvas.At(x, y) = color;
                        if (F > 0) F += dy;
                        else {
                            y--;
                            F += dydx;
                        }
                    }
                }
            }
            else
            {
                if (k > 0)
                {
                    int dydx = dy - dx, F = dy / 2 - dx;
                    int x = x_start;
                    //printf("haha1 %f\n", k);
                    for (int y = y_start; y <= y_finish;y++)
                    {
                        canvas.At(x, y) = color;
                        if (F > 0) F -= dx;
                        else
                        {
                            x++;
                            F += dydx;
                        }
                    }
                }
                else
                {
                    /*printf("haha1 %f\n", k);*/
                    int dydx = dy + dx, F = dy / 2 + dx;
                    int x = x_start;
                    for (int y = y_start; y != y_finish-1; y--) {
                        canvas.At(x, y) = color;
                        if (F < 0) F += dx;
                        else {
                            x++;
                            F += dydx;
                        }
                    }
                }
            }


        }
    }

    /******************* 5. Triangle Drawing *****************/


    bool my_p_cmpx(const glm::ivec2 & a, const glm::ivec2 & b) {
        return a.x < b.x;
    }
    void DrawTriangleFilled(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1,
        glm::ivec2 const p2) {
        // your code here:
        glm::ivec2 p_lstx[3] = { p0, p1, p2 };
        std::sort(p_lstx, p_lstx + 3, my_p_cmpx);
        
        int min_x = p_lstx[0].x;
        int max_x = p_lstx[2].x;
        
        DrawLine(canvas, color, p0, p1);
        DrawLine(canvas, color, p0, p2);
        DrawLine(canvas, color, p1, p2);

        float y1      = p_lstx[0].y;
        float y2      = p_lstx[0].y;
        float k01     = ((float) p_lstx[1].x - (float) p_lstx[0].x == 0) ? 1000 : (((float) p_lstx[1].y - (float) p_lstx[0].y) / ((float) p_lstx[1].x - (float) p_lstx[0].x));
        float k12     = ((float) p_lstx[1].x - (float) p_lstx[2].x == 0) ? 1000 : (((float) p_lstx[1].y - (float) p_lstx[2].y) / ((float) p_lstx[1].x - (float) p_lstx[2].x));
        float k02     = ((float) p_lstx[2].x - (float) p_lstx[0].x == 0) ? 1000 : (((float) p_lstx[2].y - (float) p_lstx[0].y) / ((float) p_lstx[2].x - (float) p_lstx[0].x));
        for (int i = min_x; i <= p_lstx[1].x; i++)
        {
            for (int j = std::min(y1, y2); j <= std::max(y1, y2);j++)
            {

                canvas.At(i, j) = color;
            }
            y1 += k01;
            y2 += k02;
        }
        y1 = p_lstx[2].y;
        y2 = p_lstx[2].y;
        for (int i = max_x; i >= p_lstx[1].x; i--) {
            for (int j = std::min(y1, y2); j <= std::max(y1, y2); j++) {
                canvas.At(i, j) = color;
            }
            y1 -= k02;
            y2 -= k12;
        }
    }

    /******************* 6. Image Supersampling *****************/
    void Supersample(
        ImageRGB &       output,
        ImageRGB const & input,
        int              rate) {
        // your code here:
        int x_rate = input.GetSizeX() / output.GetSizeX();
        int y_rate = input.GetSizeY() / output.GetSizeY();

        for (int i = 0; i < output.GetSizeX(); i++)
        {
            for (int j = 0; j < output.GetSizeY(); j++)
            {
                glm::vec3 color_ave = { 0.0, 0.0, 0.0 };
                int       cnt       = 0;
                for (int i_x = 0; i_x < rate; i_x++)
                {
                    for (int i_y = 0; i_y < rate; i_y++)
                    {
                        int samplex = (i+4) * x_rate + i_x;
                        int sampley = (j+4) * y_rate + i_y;
                        if (samplex >= input.GetSizeX() || sampley >= input.GetSizeY()) continue;
                        cnt++;
                        color_ave += input.At(samplex, sampley);
                    }
                }
                color_ave       = { color_ave.r / (float) cnt, color_ave.g / (float) cnt, color_ave.b / (float) cnt };
                output.At(i, j) = color_ave;
            }
        }
    }

    /******************* 7. Bezier Curve *****************/
    // Note: Please finish the function [DrawLine] before trying this part.
    glm::vec2 CalculateBezierPoint(
        std::span<glm::vec2> points,
        float const          t) {
        // your code here:
        int ori_points_num = 0;

        std::vector<glm::vec2> points_vec;
        for (auto i = points.begin(); i != points.end(); i++)
        {
            points_vec.push_back(*i);
            ori_points_num++;
        }
        for (int i = 1; i < ori_points_num; i++)
        {
            for (int j = 0; j < ori_points_num - i; j++)
            {
                points_vec[j] = (1 - t) * points_vec[j] + t * points_vec[j + 1];
            }
        }
        return points_vec[0];
    }
} // namespace VCX::Labs::Drawing2D