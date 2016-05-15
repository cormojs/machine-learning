{-# LANGUAGE BangPatterns #-}
module Main where

import Control.Monad (replicateM)
import LogisticRegression as LR
import NeuralNetwork as NN
import Data.Binary.Get

import Data.List (foldl')

import Data.Array.Repa as Repa hiding (map)
import qualified Data.Array.Repa.Operators.Mapping as Repa

import qualified Data.ByteString.Lazy as BSL

main :: IO ()
main = test2

test1 :: IO ()
test1= do
  dataset <- replicateM 100 $ do
    [s0, s1, s2, s3, name] <- words <$> getLine
    return $ ([1.0, read s0, read s1, read s2, read s3], conv name)
  let xs = map fst dataset
      ys = map snd dataset
  let thetas = (iterate (LR.gradDescent defaultAlpha xs ys) initThetas) !! 1000
  print $ h thetas [1.0, 5.0, 3.3, 1.4, 0.2]
  print $ h thetas [1.0, 5.0, 3.0, 5.1, 1.8]
  print $ h thetas [1.0, 4.4, 2.9, 1.4, 0.2]
  where conv "Iris-setosa" = 1
        conv _             = 0


test2 :: IO ()
test2 = do
  theta1 <- createRandomMatrix 100 (28 * 28 + 1)
  theta2 <- createRandomMatrix 10 101
  let nn = NN [theta1, theta2]
  images <- loadImages "train-images-idx3-ubyte"
  labels <- loadLabels "train-labels-idx1-ubyte"

  let nn' = foldl (\nn xy -> (iterate (\nn -> gradDescentS 0.01 nn xy) nn)!!1) nn (take 6000 $! zip images labels)

  images' <- loadImages "t10k-images-idx3-ubyte"
  labels' <- loadLabels "t10k-labels-idx1-ubyte"
  print $ (fromIntegral $ count 0 nn' (zip images' labels')) / (fromIntegral $ length images')
  where activate :: Double -> Int
        activate x | x > 0.5   = 1
                   | otherwise = 0
        out nn x = map activate $ toList $ computeUnboxedS $ removeBias $ last $ outputs nn x
        conv y = toList $ computeUnboxedS $ Repa.map round y
        count n _ [] = n
        count n nn ((x, y):xys) | out nn x == conv y = count (n+1) nn xys
                                | otherwise = count n nn xys

loadImages :: FilePath -> IO [Vector Repa.U]
loadImages path = do
  input <- BSL.readFile path
  return $ runGet parse input
  where parse :: Get [Vector Repa.U]
        parse = do
          getWord32be -- magic number
          n <- fromIntegral <$> getWord32be -- No. images
          getWord8    -- No. rows
          getWord8    -- No. columns
          replicateM n $ do
            !arr <- replicateM (28*28) $ (/255.0) <$> fromIntegral <$> getWord8
            return $! fromListUnboxed (Z :. (28*28) :. 1) arr

loadLabels :: FilePath -> IO [Vector Repa.U]
loadLabels path = do
  input <- BSL.readFile path
  return $ runGet parse input
  where parse = do
          getWord32be -- magic number
          n <- fromIntegral <$> getWord32be -- No. items
          replicateM n $ do
            !n <- fromIntegral <$> getWord8
            return $! vecs !! n
        vecs = [ fromListUnboxed (Z :. 10 :. 1) [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
               , fromListUnboxed (Z :. 10 :. 1) [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
               ]
