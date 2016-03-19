(ns mario-net.image
  (:require [mikera.image.core :as imagez]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [mikera.vectorz.core]
            [cortex.description :as desc]
            [cortex.layers :as layers]
            [cortex.network :as net]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [cortex.protocols :as cp]
            [clojure.core.matrix.macros :refer [c-for]])
  (:import [java.awt.image BufferedImage])
  (:gen-class))



(defn load-main-image [] (imagez/load-image (io/resource "1-1.png")))

(defonce main-image (memoize load-main-image))


(defn get-pixel-indexes
  [^BufferedImage img]
  (let [^ints pixels (imagez/get-pixels img)
        data-set (set (seq pixels))]
    (vec (map-indexed vector data-set))))

(defn diff-squared
  ^double [lhs rhs]
  (let [tmp (double (- lhs rhs))]
    (* tmp tmp)))


(defn image-to-indexed-column-major-matrix
  [^BufferedImage img]
  (let [pixel-indexes (get-pixel-indexes img)
        pixel-to-index-map (into {} (map (comp vec reverse) pixel-indexes))
        ^ints pixels (imagez/get-pixels img)
        indexes (mapv #(pixel-to-index-map %) pixels)
        mean (double (/ (m/esum indexes) (count indexes)))
        deviation (double (/ (reduce + (map #(diff-squared mean %) indexes))
                          (count indexes)))
        indexes (map #(double (/ (- % mean) deviation)) indexes)
        width (.getWidth img)
        height (.getHeight img)]
    {:mean mean :deviation deviation
     :pixel-indexes pixel-indexes
     :data (m/pack (m/transpose (m/array :vectorz (mapv vec (partition width indexes)))))}))


(defn matrix-to-input-vectors
  [mat row-width]
  (let [last-row (- (m/row-count mat) row-width)
        mat-as-vec (m/as-vector mat)
        column-stride (m/column-count mat)
        row-width-stride (* row-width column-stride)]
    (mapv (fn [idx]
            (m/pack
             (m/subvector mat-as-vec (* idx column-stride) row-width-stride)))
          (range last-row))))


(defn image-to-mean-stddev-and-input-vectors
  [^BufferedImage img]
  (let [{:keys [mean deviation data] :as matrix-data}
        (image-to-indexed-column-major-matrix img)
        input-vecs (matrix-to-input-vectors data 5)]
    (assoc matrix-data :input-vectors input-vecs)))

(defn clamp
  [val min-val max-val]
  (max (min val max-val) min-val))


(defn mat-to-image
  [matrix {:keys [mean deviation pixel-indexes]}]
  (let [num-indexes (count pixel-indexes)
        index-map (into {} pixel-indexes)
        new-mat (m/transpose matrix)
        pixels (int-array (map (fn [val]
                                 (index-map (int (clamp (Math/round
                                                         (+ mean (* val deviation)))
                                                        0
                                                        (- num-indexes 1)))))
                               (m/eseq new-mat)))
        retval (imagez/new-image (m/column-count new-mat) (m/row-count new-mat))]
    (imagez/set-pixels retval pixels)
    retval))


(def autoencoder-layer-sizes [256 128 64 32 16])


(defn train-autoencoder-layer
  [input layer-size]
  (let [input-size (m/ecount (first input))
        network (desc/build-and-create-network [(desc/input input-size)
                                                (desc/gaussian-noise)
                                                (desc/linear->logistic layer-size)
                                                (desc/linear input-size)])
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        loss-fn (opt/mse-loss)
        network (net/train
                 network optimizer loss-fn
                 input input 5 10 input input)
        ;;drop the noise and the decoder
        min-network (assoc network :modules (vec (drop 1 (drop-last (:modules network)))))
        next-data (net/run min-network input)
        modules (:modules network)]
    {:encoder {:weights (:weights (second modules)) :bias (:bias (second modules))}
     :decoder {:weights (:weights (last  modules)) :bias (:bias (last  modules))}
     :next-input next-data}))


(defn layers->full-network
  [layers input-size with-noise]
  (let [network-input (if with-noise
                        [(desc/input input-size)
                         (desc/gaussian-noise)]
                        [(desc/input input-size)])]
   (desc/build-and-create-network
    (concat network-input
            (map (fn [layer]
                   (let [weights (get-in layer [:encoder :weights])
                         bias (get-in layer [:encoder :bias])
                         n-input (m/column-count weights)
                         n-output (m/row-count weights)]
                     (desc/linear->logistic n-output
                                            :weights weights :bias bias)))
                 layers)
            (map (fn [layer]
                   (let [weights (get-in layer [:decoder :weights])
                         bias (get-in layer [:decoder :bias])
                         n-output (m/row-count weights)]
                     (desc/linear n-output :weights weights :bias bias)))
                 (reverse layers))))))

(defn train-full-encoder-decoder-network
  [input layers]
  (let [input-size (m/ecount (first input))
        loss-fn (opt/mse-loss)
        network (layers->full-network layers input-size true)
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        network (net/train
                 network optimizer loss-fn
                 input input 5 10 input input)
        front-half (* 2 (count layers))
        ;;drop the noise layer
        modules (drop 1 (:modules network))
        encoder-layers (mapcat drop-last
                               (partition
                                2
                                (take front-half modules)))
        decoder-layers (reverse (drop front-half modules))]
    (mapv (fn [encoder decoder]
            {:encoder {:weights (:weights encoder) :bias (:bias encoder)}
             :decoder {:weights (:weights decoder) :bias (:bias decoder)}})
          encoder-layers decoder-layers)))


(def stacked-network (atom nil))
(def full-network (atom nil))



(defn train-autoencoder-network
  ([data layer-sizes]
   (let [layers (reduce (fn [layers layer-size]
                          (let [input-data (or (:next-input (last layers))
                                               data)]
                            (conj layers (train-autoencoder-layer input-data layer-size))))
                        []
                        layer-sizes)
         _ (reset! stacked-network layers)
         layers (train-full-encoder-decoder-network data layers)]
     (reset! full-network layers)
     layers))

  ([] (let [scaled-image (imagez/scale (main-image) 0.25)
            data (image-to-mean-stddev-and-input-vectors scaled-image)
            layers
            (train-autoencoder-network (:input-vectors data) autoencoder-layer-sizes)]
        (assoc data :layers layers :scaled-image scaled-image))))


;;render the right-most column, copy into the data store and render the next
;;column
(defn render-image-from-autoencoder-network
  [{:keys [mean variance data layers scaled-image pixel-indexes] :as data-set}]
  (let [row-width 5
        mat data
        column-stride (m/column-count mat)
        row-width-stride (* row-width column-stride)
        data-vector (m/new-array :vectorz (* row-width column-stride))
        initial-data-vector (m/subvector data-vector 0 (* (dec row-width) column-stride))
        _ (m/assign! initial-data-vector
                     (m/subvector (m/as-vector data) 0 (* (dec row-width))))
        output-mat (m/new-array :vectors (m/shape data))
        _ (m/assign! (m/subvector output-mat 0 (* (dec row-width) column-stride))
                     (m/subvector (m/as-vector data) 0 (* (dec row-width))))
        network (layers->full-network layers false)
        columns-to-render (- (m/row-count data) row-width)]
    ;;generate columns
    (doseq [idx (range columns-to-render)]
      (let [network (cp/forward network data-vector)
            output (cp/output network)]
        (m/assign! initial-data-vector
                   (m/subvector output column-stride (* (dec row-width) column-stride)))
        (m/assign! (m/subvector (m/as-vector output-mat)
                                (* (+ idx (dec row-width)) column-stride)
                                column-stride))))
    (mat-to-image output-mat data-set)))


(defn image-to-indexed-squares
  [^BufferedImage src-image ^long square-size]
  (let [input-width (.getWidth src-image)
        input-height (.getHeight src-image)
        output-width (quot input-width square-size)
        output-height (quot input-height square-size)
        ^ints input-data (imagez/get-pixels src-image)
        output-pixel-count (* output-width output-height)
        retval (reduce (fn [[index-map output-pixels] ^long output-idx]
                         (let [output-y (quot output-idx output-width)
                               output-x (rem output-idx output-width)
                               input-y (* output-y square-size)
                               input-x (* output-x square-size)
                               input-values (vec (for [y (range square-size)
                                                       x (range square-size)]
                                                   (let [input-offset (+ (* (+ input-y y)
                                                                            input-width)
                                                                         (+ input-x x))]
                                                     (aget input-data input-offset))))
                               item-idx (get index-map input-values (count index-map))]
                           [(assoc index-map input-values item-idx)
                            (conj output-pixels item-idx)]))
                       [{} []]
                       (range output-pixel-count))
        [item-map output-pixels] retval
        output-ary (m/array :vectorz (mapv vec (partition output-width output-pixels)))
        ]

    {:item-map item-map :pixels output-ary}))


(defn indexed-squares-to-image
  [{:keys [item-map pixels]}]
  (let [input-square-count (count (ffirst item-map))
        square-size (int (Math/sqrt input-square-count))
        reverse-map (into {} (map (comp vec reverse) item-map))
        output-pixel-count (long (m/ecount pixels))
        input-pixel-count (* output-pixel-count square-size square-size)
        input-width (long (* square-size (m/column-count pixels)))
        retval (imagez/new-image input-width
                                 (* square-size (m/row-count pixels)))
        ^ints input-pixels (imagez/get-pixels retval)
        output-width (long (m/column-count pixels))
        output-pixel-vec (m/as-vector pixels)]
    (c-for
     [output-pixel 0 (< output-pixel output-pixel-count) (inc output-pixel)]
     (let [output-x (rem output-pixel output-width)
           output-y (quot output-pixel output-width)
           input-x (* output-x square-size)
           input-y (* output-y square-size)
           output-pixel (int (m/mget output-pixel-vec output-pixel))
           input-data (reverse-map output-pixel)]
       (c-for
        [y 0 (< y square-size) (inc y)]
        (c-for
         [x 0 (< x square-size) (inc x)]
         (let [input-val (input-data (+ (* y square-size) x))
               input-addr (+ (* (+ input-y y) input-width) (+ input-x x))]
           (aset input-pixels input-addr input-val))))))
    (imagez/set-pixels retval input-pixels)
    retval))

(def column-data-count 2)


(defn build-square-training-data
  "Build a dataset that given two columns and an index, output the item in
the third column"
  [square-size]
  (let [indexed-data (image-to-indexed-squares (main-image) square-size)
        column-data (m/transpose (:pixels indexed-data))
        output-count (count (:item-map indexed-data))
        column-height (m/column-count column-data)
        num-columns (m/row-count column-data)

        zero-vec (vec (repeat output-count 0.0))
        mat-vec (m/as-vector column-data)
        valid-column-indexes (- num-columns column-data-count)
        block-size (* column-data-count column-height)
        data-label-seq (mapcat identity
                               (for [col-idx (range valid-column-indexes)]
                                 (let [data-vec (m/subvector mat-vec (* col-idx column-height)
                                                             block-size)]
                                   (for [idx (range column-height)]
                                     [(m/array :vectorz (concat (m/eseq data-vec) [idx]))
                                      (m/array :vectorz (assoc zero-vec
                                                               (int (m/mget
                                                                     data-vec
                                                                     (+ (*
                                                                         (- column-data-count 1)
                                                                         column-height) idx)))
                                                               1.0))]))))
        training-data (mapv first data-label-seq)
        training-labels (mapv second data-label-seq)]
    {:indexed-squares indexed-data :column-data column-data
     :training-data training-data :training-labels training-labels
     :data-and-labels data-label-seq}))


(defn even-training-data-distribution
  [training-data training-labels item-sample-count]
  (let [groups (group-by second (map vector training-data training-labels))
        data-and-labels (mapcat identity (for [[key item-vec] groups]
                                           (let [item-count (count item-vec)
                                                 indexes (vec
                                                          (shuffle (range (count item-vec))))]
                                             (for [idx (range item-sample-count)]
                                               (let [local-idx (rem idx item-count)]
                                                 (item-vec (indexes local-idx)))))))
        training-data (mapv first data-and-labels)
        training-labels (mapv second data-and-labels)]
    [training-data training-labels]))


(defn create-network-scaler
  [square-data]
  (let [indexed-squares (:indexed-squares square-data)
        item-count (count (:item-map indexed-squares))
        column-count (long (m/column-count (:column-data square-data)))]
    (m/array :vectorz (concat (repeat (* column-count column-data-count)
                                      (/ 1.0 item-count))
                              [(/ 1.0 column-count)]))))


(defn train-square-training-network
  [square-size]
  (let [square-data (build-square-training-data square-size)
        output-size (m/ecount (first (:training-labels square-data)))
        input-size (m/ecount (first (:training-data square-data)))
        data-size (count (:training-labels square-data))
        indexes (shuffle (range data-size))
        training-count (int (* 0.8 data-size))
        training-indexes (take training-count indexes)
        cv-indexes (drop training-count indexes)
        network-scale (create-network-scaler square-data)
        all-training-data (mapv #(m/mul % network-scale) (:training-data square-data))
        training-data (mapv all-training-data training-indexes)
        training-labels (mapv (:training-labels square-data) training-indexes)
        [training-data training-labels] (even-training-data-distribution
                                         training-data training-labels 1000)
        _ (println (count training-data) (first training-data))
        cv-data (mapv all-training-data cv-indexes)
        cv-labels (mapv (:training-labels square-data) cv-indexes)
        layer-sizes [200 100]
        network (desc/build-and-create-network [(desc/input input-size)
                                                (mapcat (fn [layer-size]
                                                          (desc/linear->logistic layer-size))
                                                        layer-sizes)
                                                (desc/softmax output-size)])
        loss-fn (opt/->CrossEntropyLoss)
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        network (net/train network optimizer loss-fn
                           training-data training-labels
                           5 20
                           cv-data cv-labels)]
    {:network network :square-data square-data}))


(defn network-output-to-index
  [output-vec]
  (let [num-output (long (m/ecount output-vec))
        last-valid (- num-output 1)]
    (loop [max-prob 0.0
           max-idx 0
           idx 0]
      (if (< idx num-output)
        (let [prob (m/mget output-vec idx)
              [max-prob max-idx] (if (> prob max-prob)
                                   [prob idx]
                                   [max-prob max-idx])]

          (recur max-prob max-idx (inc idx)))
        [max-idx output-vec]))))

(defn check-data-and-label
  [data label]
  (let [elem-count (long (m/ecount data))
        col-count (quot elem-count
                        2)
        index-idx (- elem-count 1)
        index (int (m/mget data index-idx))
        answer (double (first (network-output-to-index label)))
        value (m/mget data (+ col-count index))]
    (= answer value)))

(defn verify-training-data
  [training-data training-labels]
  (let [data-and-labels (mapv vector training-data training-labels)]
    (remove (fn [[data label]]
              (check-data-and-label data label))
            data-and-labels)))


(defn draw-network-square
  [input-vec idx network]
  (let [real-input (m/array :vectorz (concat (m/eseq input-vec) [idx]))
        network (cp/forward network real-input)]
    (first (network-output-to-index (cp/output network)))))


(defn draw-network-column
  [input-vec scale-vec column-height network]
  (let [real-input (m/array :vectorz (concat (m/eseq input-vec) [0.0]))
        index-idx (- (m/ecount real-input) 1)
        column-data (for [col (range column-height)]
                      (do
                        (m/mset! real-input index-idx (double col))
                        (let [network (cp/forward network (m/mul real-input scale-vec))]
                          (first (network-output-to-index (cp/output network))))))]
    (vec column-data)))


(defn draw-image
  [network-and-data column-count]
  (let [network (:network network-and-data)
        square-data (:square-data network-and-data)
        {:keys [indexed-squares column-data training-data training-labels]} square-data
        input-vec (m/array :vectorz (drop-last (m/eseq
                                                (first training-data))))
        column-height (m/column-count column-data)
        scale-vec (create-network-scaler square-data)
        results (mapv vec (partition column-height (m/eseq input-vec)))
        results
        (reduce (fn [results col-idx]
                  (let [new-column (draw-network-column input-vec
                                                        scale-vec
                                                        column-height network)]
                    (m/assign! (m/subvector input-vec 0 column-height)
                               (m/subvector input-vec column-height column-height))
                    (m/assign! (m/subvector input-vec column-height column-height)
                               new-column)
                    (conj results new-column)))
                results
                (range column-count))
        output-mat (m/transpose (m/array :vectorz results))]
    (indexed-squares-to-image (assoc indexed-squares :pixels output-mat))))


(defn -main
  [& args]
  (train-autoencoder-network))
