using Metalhead, Test
using Flux


@testset "MLPMixer" begin
<<<<<<< HEAD
    @test size(MLPMixer()(rand(Float32, 256, 256, 3, 2))) == (1000, 2)
    @test_skip gradtest(MLPMixer(), rand(Float32, 256, 256, 3, 2))
end

@testset "ESRGAN" begin
    esrgan = ESRGAN()
    D = esrgan[:discrimator]
    G = esrgan[:generator]
    @test size(G(rand(Float32,24,24,3,5))) == (96, 96, 3, 5)
    @test size(D(G(rand(Float32,24,24,3,5)))) == (1,5)
end
=======
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = MLPMixer(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
      GC.gc()
    end
  end
end

@testset "ResMLP" begin
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = ResMLP(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 1))
      GC.gc()
    end
  end
end

@testset "gMLP" begin
  @testset for mode in [:small, :base, :large, :huge]
    @testset for drop_path_rate in [0.0, 0.5, 0.99]
      m = gMLP(mode; drop_path_rate)
      @test size(m(rand(Float32, 224, 224, 3, 2))) == (1000, 2)
      @test_skip gradtest(m, rand(Float32, 224, 224, 3, 2))
      GC.gc()
    end
  end
end
>>>>>>> aba6fb832093d88dc2d2b4d5b1d2d63a0f21eb9c
