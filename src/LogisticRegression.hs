module LogisticRegression where


gradDescent :: Double -> [[Double]] -> [Double] -> [Double] -> [Double]
gradDescent alpha xs ys thetas = map update $ zip [0..] thetas
  where update (j, theta) = theta - alpha * ((sum $ map (s j) [0..m-1]) / fromIntegral m)
        s j i = ((h thetas $ xs !! i) - (ys !! i)) * (xs !! i !! j)
        m = length xs

h :: [Double] -> [Double] -> Double
h thetas x = 1.0 / (1.0 + exp (-1 * (thetas `dot` x)))

dot :: [Double] -> [Double] -> Double
dot [] [] = 0.0
dot (a:as) (b:bs) = a * b + dot as bs

defaultAlpha :: Double
defaultAlpha = 0.2

initThetas :: [Double]
initThetas = [1.0, 1.0, 1.0, 1.0, 1.0]
